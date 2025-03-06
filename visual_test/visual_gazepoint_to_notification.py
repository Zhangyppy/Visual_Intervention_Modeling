# import gym
# from gym import Env
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pickle
import time
import pygame
import numpy as np
import random
import os
import torch as th
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO

# Pygame settings
pygame.init()
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Constants
AGENT_RADIUS = 20
BLOCK_SIZE = 80
MOVE_SPEED = 20

# Check if CUDA or MPS is available
device = (
    "cuda"
    if th.cuda.is_available()
    else "mps" if th.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")


class VisualSimulationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(VisualSimulationEnv, self).__init__()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        # --- ACTION SPACE ---
        # We will choose from 4 discrete directions:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # --- OBSERVATION SPACE ---
        # The observation is a dictionary containing both the visual observation and position information
        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        SCREEN_HEIGHT,
                        SCREEN_WIDTH,
                        3,
                    ),  # Full screen (height, width, channels)
                    dtype=np.uint8,
                ),
                "position": spaces.Box(
                    low=np.array([0, 0], dtype=np.float32),
                    high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT], dtype=np.float32),
                    shape=(2,),  # Agent's x, y position
                    dtype=np.float32,
                ),
                # "action_history": spaces.Box(
                #     low=0,
                #     high=255,
                #     shape=(10,),
                #     dtype=np.uint8,
                # ),
            }
        )

        # --- Episode Variables ---
        self._ep_length = 100  # Max episode length
        self._steps = 0  # Current step

        # Initialize positions of agent and target
        self._initialize_positions()

        # Initialize pygame if in human render mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            self._init_render()

        self.reward = 0
        self.reward_count = 0
        self.exploration_count = 0  # Track how many exploration rewards/penalties given
        self.action_history = np.full(
            10, 255, dtype=np.uint8
        )  # Reset action history with 255 (no action)

    def _initialize_positions(self):
        """Initialize agent and target positions"""
        self.agent_pos = np.array(
            [
                random.randint(AGENT_RADIUS, SCREEN_WIDTH - AGENT_RADIUS),
                random.randint(AGENT_RADIUS, SCREEN_HEIGHT - AGENT_RADIUS),
            ],
            dtype=np.float32,
        )

        # Place the target at a random position, but ensure it's not right on top of the agent
        while True:
            self.target_pos = np.array(
                [
                    random.randint(BLOCK_SIZE // 2, SCREEN_WIDTH - BLOCK_SIZE // 2),
                    random.randint(BLOCK_SIZE // 2, SCREEN_HEIGHT - BLOCK_SIZE // 2),
                ],
                dtype=np.float32,
            )

            # Make sure the target is not too close to the agent at initialization
            if np.linalg.norm(self.agent_pos - self.target_pos) > 150:
                break

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Visual Attention Task")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset positions
        self._initialize_positions()

        self.reward = 0
        self._steps = 0
        self.exploration_count = 0
        self.reward_count = 0
        self.action_history = np.full(
            10, 255, dtype=np.uint8
        )  # Reset action history with 255 (no action)

        # Process and handle events to avoid pygame becoming unresponsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        return self._get_obs(), {}

    def _get_target_boundaries(self):
        """
        Calculate the target boundaries
        Returns:
            tuple: (x_min, y_min, x_max, y_max) of target boundaries
        """
        x_min = int(self.target_pos[0] - BLOCK_SIZE // 2)
        y_min = int(self.target_pos[1] - BLOCK_SIZE // 2)
        x_max = x_min + BLOCK_SIZE
        y_max = y_min + BLOCK_SIZE
        return x_min, y_min, x_max, y_max

    def _check_collision(self):
        """Check if the agent collides with the target"""
        # Calculate the center of the target
        target_center = self.target_pos

        # Check if the distance between the agent's center and the target's center
        # is less than the sum of the agent's radius and half the target size
        distance = np.linalg.norm(self.agent_pos - target_center)
        return distance < (AGENT_RADIUS + BLOCK_SIZE / 2)

    def _detect_oscillation(self, action):
        """Detect if the agent is oscillating between actions"""
        # handle init case
        if self._steps < 4:
            return False

        # Check for simple back-and-forth pattern (e.g., 0-1-0-1)
        # Require at least 4 actions in the pattern to avoid false positives
        if (
            len(set(self.action_history[:4])) <= 2  # At most 2 unique actions
            and self.action_history[0] == self.action_history[2]
            and self.action_history[1] == self.action_history[3]
        ):
            return True

        # Check for repetitive single action (allow up to 3 of the same action - agent might need to move far)
        if len(set(self.action_history[:3])) == 1:
            return True

        return False

    def step(self, action):
        # Process events to prevent pygame from becoming unresponsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        self._steps += 1
        done = False

        # Update action history
        self.action_history = np.roll(self.action_history, 1)
        self.action_history[0] = action

        # Move the agent
        action_map = {
            0: np.array([0, -MOVE_SPEED]),  # Up
            1: np.array([MOVE_SPEED, 0]),  # Right
            2: np.array([0, MOVE_SPEED]),  # Down
            3: np.array([-MOVE_SPEED, 0]),  # Left
        }

        # Store the previous position
        prev_pos = self.agent_pos.copy()

        # Calculate the intended (unclipped) position
        intended_pos = self.agent_pos + action_map[action]

        # Check if the intended position would be outside the screen boundaries
        hit_boundary = (
            intended_pos[0] <= AGENT_RADIUS
            or intended_pos[0] >= SCREEN_WIDTH - AGENT_RADIUS
            or intended_pos[1] <= AGENT_RADIUS
            or intended_pos[1] >= SCREEN_HEIGHT - AGENT_RADIUS
        )

        # Clip the agent position to the screen boundaries
        self.agent_pos = np.clip(
            intended_pos,
            [AGENT_RADIUS, AGENT_RADIUS],
            [SCREEN_WIDTH - AGENT_RADIUS, SCREEN_HEIGHT - AGENT_RADIUS],
        )

        # Calculate movement distance
        # move_distance = np.linalg.norm(self.agent_pos - prev_pos)

        # --- Reward Structure ---
        reward = 0

        # Boundary penalty
        if hit_boundary:
            reward -= 10

        # Terminal reward for reaching target
        if self._check_collision():
            reward += 200  # Big reward for reaching target
            self.reward_count += 1
            if self.reward_count >= 3:
                done = True

        # Reward for exploring (changing actions)
        # Check for oscillation patterns in action history
        if not self._detect_oscillation(action):
            # Diminishing exploration reward based on steps taken
            exploration_reward = 0.3 / (1 + self.exploration_count * 0.1)
            # Penalize if taking too many steps without reaching target
            if (
                self._steps > self._ep_length * 0.5
            ):  # If more than 50% of episode length
                exploration_reward -= 0.3 * (self._steps / self._ep_length)
            reward += exploration_reward
            self.exploration_count += 1
        else:
            reward -= 0.5

        # Check if episode is done due to step limit
        if self._steps >= self._ep_length:
            done = True

        # Store the reward and update action history
        self.reward += reward

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create a canvas for the full screen observation
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(WHITE)

        # Draw the blue target block
        target_x_min, target_y_min, _, _ = self._get_target_boundaries()
        pygame.draw.rect(
            canvas, BLUE, (target_x_min, target_y_min, BLOCK_SIZE, BLOCK_SIZE)
        )

        # Draw the red agent
        pygame.draw.circle(canvas, RED, self.agent_pos.astype(int), AGENT_RADIUS)

        # draw the space boundaries outer boundary
        pygame.draw.rect(canvas, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        # Convert the canvas to a numpy array
        visual_obs = pygame.surfarray.array3d(canvas)
        # PyGame returns in (W, H, C) format, convert to (H, W, C)
        visual_obs = visual_obs.transpose(1, 0, 2)

        # Return both visual observation and position information
        return {
            "visual": visual_obs,
            "position": self.agent_pos.astype(np.float32),
            "action_history": self.action_history,
        }

    def render(self, mode="human"):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if mode == "rgb_array":
            # For rgb_array mode, return the full screen view
            return self._render_frame()

        elif mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                pygame.display.set_caption("Visual Attention Task")
                self.clock = pygame.time.Clock()

            # Render the full frame
            self._render_frame()

            # Update the display
            pygame.display.flip()
            self.clock.tick(30)

            # Process events to prevent pygame from becoming unresponsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

    def _render_frame(self):
        """Render the full environment to the screen"""
        # Fill the background
        self.screen.fill(WHITE)

        # Draw the blue target block
        target_x_min, target_y_min, _, _ = self._get_target_boundaries()
        pygame.draw.rect(
            self.screen, BLUE, (target_x_min, target_y_min, BLOCK_SIZE, BLOCK_SIZE)
        )

        # Draw the red agent
        pygame.draw.circle(self.screen, RED, self.agent_pos.astype(int), AGENT_RADIUS)

        # draw the space boundaries outer boundary
        pygame.draw.rect(self.screen, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        # Draw text info
        font = pygame.font.SysFont("Arial", 14)

        # Display step count
        steps_text = font.render(f"Steps: {self._steps}/{self._ep_length}", True, BLACK)
        self.screen.blit(steps_text, (10, 10))

        # Display distance
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        distance_text = font.render(f"Distance: {distance:.1f}", True, BLACK)
        self.screen.blit(distance_text, (10, 30))

        # Display reward
        reward = self.reward
        reward_text = font.render(f"Cumulative Reward: {reward:.2f}", True, BLACK)
        self.screen.blit(reward_text, (10, 50))

        # Display last four actions and exploration info
        action_names = {
            0: "Up",
            1: "Right",
            2: "Down",
            3: "Left",
            None: "None",
            255: "None",
        }

        # Get the last 4 actions from action history if available
        action_history = []
        if hasattr(self, "action_history"):
            action_history = (
                self.action_history[-4:] if len(self.action_history) > 0 else []
            )
        else:
            # Fallback to last and second last if action_history not available
            if hasattr(self, "last_action"):
                action_history.append(self.last_action)
            if hasattr(self, "second_last_action"):
                action_history.insert(0, self.second_last_action)

        # Pad with None if we have fewer than 4 actions
        while len(action_history) < 4:
            action_history.insert(0, None)

        # Convert to action names
        action_history_names = [action_names.get(a, str(a)) for a in action_history]

        # Display the action history
        history_text = font.render(
            f"Recent actions: {' → '.join(action_history_names)}",
            True,
            BLACK,
        )
        self.screen.blit(history_text, (10, 70))

        # Display exploration count
        exploration_text = font.render(
            f"Explored: {self.exploration_count}",
            True,
            BLACK,
        )
        self.screen.blit(exploration_text, (10, 90))

        # Return the screen as a numpy array if needed
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        # Initialize with the visual part of the observation space
        super().__init__(observation_space, features_dim)

        # Extract shapes from the observation space
        n_input_channels = observation_space["visual"].shape[
            0
        ]  # Channels are first in CHW format
        pos_dim = observation_space["position"].shape[0]
        # action_history_dim = observation_space["action_history"].shape[0]

        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(
                n_input_channels, 32, 8, stride=4
            ),  # Larger kernel for full-screen input
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 4, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 128, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Calculate the size of the flattened features
        with th.no_grad():
            # Process a sample visual observation (already in CHW format)
            sample = th.as_tensor(observation_space["visual"].sample()[None]).float()
            # print(f"Sample shape: {sample.shape}")  # Should be (1, C, H, W)
            n_flatten = self.cnn(sample).shape[1]
            # print(f"Flattened features size: {n_flatten}")

        # Create a network for processing position data
        self.pos_net = th.nn.Sequential(th.nn.Linear(pos_dim, 64), th.nn.ReLU())

        # Create a network for processing action history
        # self.action_history_net = th.nn.Sequential(
        #     th.nn.Linear(action_history_dim, 32), th.nn.ReLU()
        # )

        # Combine visual, position and action history features
        self.combined = th.nn.Sequential(
            th.nn.Linear(n_flatten + 64, features_dim),
            th.nn.LayerNorm(features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        # Process visual features (already in CHW format from CustomVecTranspose)
        visual_obs = th.as_tensor(observations["visual"]).float()

        # Add batch dimension (CHW -> NCHW)
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        visual_features = self.cnn(visual_obs)

        # Process position features
        pos_obs = th.as_tensor(observations["position"]).float()

        # Add batch dimension if needed
        if pos_obs.dim() == 1:
            pos_obs = pos_obs.unsqueeze(0)  # (2,) -> (1,2)
        pos_features = self.pos_net(pos_obs)

        # Process action history features
        # action_history_obs = th.as_tensor(observations["action_history"]).float()

        # Add batch dimension if needed
        # if action_history_obs.dim() == 1:
        #     action_history_obs = action_history_obs.unsqueeze(0)
        # action_history_features = self.action_history_net(action_history_obs)

        # Combine features
        combined = th.cat(
            [visual_features, pos_features], dim=1
        )

        return self.combined(combined)


class CustomVecTranspose(VecTransposeImage):
    def __init__(self, venv):
        super().__init__(venv)

        # Update observation space to handle dictionary
        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH),  # CHW format
                    dtype=np.uint8,
                ),
                "position": venv.observation_space[
                    "position"
                ],  # Keep position space as is
                # "action_history": spaces.Box(
                #     low=0, high=255, shape=(10,), dtype=np.uint8
                # ),
            }
        )

    def transpose_observations(self, obs):
        """
        Transpose observations from the vectorized environment.
        DummyVecEnv wraps observations in a list/array, so we need to handle that.
        """
        # Handle vectorized observations
        if isinstance(obs, (list, np.ndarray)):
            # For vectorized observations (e.g., from DummyVecEnv)
            visual_obs = obs[0]["visual"]  # Shape: (H, W, C) or (1, H, W, C)
            position = obs[0]["position"]  # Shape: (2,) or (1, 2)
            # action_history = obs[0]["action_history"]  # Shape: (10,) or (1, 10)
        else:
            # For single observations (e.g., during evaluation)
            visual_obs = obs["visual"]  # Shape: (H, W, C)
            position = obs["position"]  # Shape: (2,)
            # action_history = obs["action_history"]  # Shape: (10,)

        # Remove any extra batch dimension from DummyVecEnv if present
        if len(visual_obs.shape) == 4:
            visual_obs = visual_obs[0]  # Convert (1, H, W, C) to (H, W, C)
        if len(position.shape) == 2:
            position = position[0]  # Convert (1, 2) to (2,)
        # if len(action_history.shape) == 2:
        #     action_history = action_history[0]  # Convert (1, 10) to (10,)

        # Convert from HWC to CHW format
        visual_obs = np.transpose(visual_obs, (2, 0, 1))  # (H, W, C) -> (C, H, W)

        return {
            "visual": visual_obs,
            "position": position,
            # "action_history": action_history,
        }


def save_checkpoint(
    model, timestep, checkpoint_dir="Training/Checkpoints", model_tag=None
):
    """Save a checkpoint of the model"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    tag = model_tag if model_tag is not None else "interrupted"
    checkpoint_path = os.path.join(
        checkpoint_dir, f"ppo_visual_attention_checkpoint_{tag}_{timestep}_steps"
    )
    model.save(checkpoint_path)
    print(f"Saved checkpoint at {checkpoint_path}")


def continue_training(model_path, additional_timesteps=100000, checkpoint_freq=10000):
    """
    Continue training a saved model for additional timesteps.

    Args:
        model_path: Path to the saved model (either a checkpoint or final model)
        additional_timesteps: Number of additional timesteps to train for
        checkpoint_freq: How often to save checkpoints during continued training
    """
    print(f"Loading model from {model_path} for continued training...")

    # Create a fresh environment
    env = make_env()

    # Load the saved model
    model = PPO.load(model_path, env=env)

    # Setup checkpoint callback
    checkpoint_dir = "Training/Checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="ppo_visual_attention_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1,
    )

    print(f"Continuing training for {additional_timesteps} more timesteps...")

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            progress_bar=True,
            callback=checkpoint_callback,
            reset_num_timesteps=False,  # Important: continue counting timesteps
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        model.save(os.path.join(checkpoint_dir, "ppo_visual_attention_interrupted"))
        print("Checkpoint saved. You can resume training later.")

    # Save the final model with a timestamp
    timestamp = int(time.time())
    final_path = f"ppo_visual_attention_continued_{timestamp}"
    model.save(final_path)
    print(f"Saved continued model to {final_path}")

    return model


# ==== Initialization ====
# Create environment pipeline
def make_env():
    env = VisualSimulationEnv()
    env = Monitor(env)
    return env


env = DummyVecEnv([make_env])
# Use custom wrapper for dictionary observation space
env = CustomVecTranspose(env)

# Configure model parameters
policy_kwargs = dict(
    features_extractor_class=VisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 64]
)

# Initialize model
model = PPO(
# model = RecurrentPPO(
       "MultiInputPolicy",
    # "MultiInputLstmPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    # n_steps=2048,
    # batch_size=64,
    n_steps=4096,
    batch_size=128,
    ent_coef=0.01,
    tensorboard_log=os.path.join("Training", "Logs"),
    device=device,
)

# ==== Training ====

# Training with checkpointing
total_timesteps = 1000000
check_freq = 25000  # Save checkpoint every 25k steps
MODEL_TAG = "StandardPPO"

# Check if there's a latest checkpoint to resume from
checkpoint_dir = "Training/Checkpoints"
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    checkpoints = [
        f
        for f in os.listdir(checkpoint_dir)
        if f.startswith(f"ppo_visual_attention_checkpoint_{MODEL_TAG}_")
    ]
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoints")
        print(f"Checkpoints: {checkpoints}")
        # Extract the timestep from filenames that end with '_NUMBER_steps.zip'
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-2]))
        latest_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint)
        print(f"Found latest checkpoint: {latest_checkpoint}")

# Load the latest checkpoint if it exists
if latest_checkpoint:
    print("Loading latest checkpoint...")
    # model = PPO.load(latest_checkpoint, env=env)
    model = RecurrentPPO.load(latest_checkpoint, env=env)
    # Extract the timestep from the checkpoint name, considering the '_steps.zip' suffix
    start_timestep = int(latest_checkpoint.split("_")[-2])
    remaining_timesteps = total_timesteps - start_timestep
    print(f"Resuming training from timestep {start_timestep}")
else:
    start_timestep = 0
    remaining_timesteps = total_timesteps

# Create the checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=check_freq,
    save_path=checkpoint_dir,
    name_prefix=f"ppo_visual_attention_checkpoint_{MODEL_TAG}",
    save_replay_buffer=False,
    save_vecnormalize=True,
    verbose=1,
)

# Start training with proper checkpointing
try:
    model.learn(
        total_timesteps=remaining_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
    )
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving final checkpoint...")
    save_checkpoint(model, start_timestep + model.num_timesteps, model_tag=MODEL_TAG)
    print("Checkpoint saved. You can resume training later.")

# Save final model
model.save("ppo_visual_attention_full")


# To continue training after the initial run completes, uncomment and run:
# continue_training("ppo_visual_attention_full", additional_timesteps=200000)

# For testing/evaluation
env = VisualSimulationEnv(render_mode="human")
env = Monitor(env)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f}")
