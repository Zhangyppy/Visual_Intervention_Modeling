import datetime
import gc
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
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
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
                # NOTE: We initialize with HWC format, which will be converted to CHW
                # by the CustomVecTranspose wrapper
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

        # Store the reward for rendering purposes
        self.reward += reward

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create a canvas for the full screen observation or reuse existing one
        if not hasattr(self, '_obs_canvas'):
            self._obs_canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # Reuse canvas to avoid memory allocation each time
        canvas = self._obs_canvas
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

        # Convert the canvas to a numpy array - reuse buffer if possible
        if not hasattr(self, '_visual_obs_buffer') or self._visual_obs_buffer.shape != (SCREEN_HEIGHT, SCREEN_WIDTH, 3):
            self._visual_obs_buffer = np.empty((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # Get pixel data into our buffer - more efficient than creating new array
        # PyGame returns in (W, H, C) format, convert to (H, W, C)
        pygame.pixelcopy.surface_to_array(np.transpose(self._visual_obs_buffer, (1, 0, 2)), canvas)
        
        # Return both visual observation and position information
        return {
            "visual": self._visual_obs_buffer,
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
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, input_channels=None):
        # Initialize with the visual part of the observation space
        super().__init__(observation_space, features_dim)

        # Extract shapes from the observation space
        # The format depends on whether the CustomVecTranspose was applied or not
        visual_space = observation_space["visual"]
        
        # If input_channels is specified, use it (for frame stacking)
        if input_channels is not None:
            n_input_channels = input_channels
            print(f"Using provided input channels: {n_input_channels}")
        else:
            # Determine the correct channel dimension
            if len(visual_space.shape) == 3:  # (C, H, W) format after CustomVecTranspose
                n_input_channels = visual_space.shape[0]
            else:  # (H, W, C) format before CustomVecTranspose
                n_input_channels = visual_space.shape[2]
            
        pos_dim = observation_space["position"].shape[0]
        # action_history_dim = observation_space["action_history"].shape[0]
        
        print(f"Initializing CNN with {n_input_channels} input channels")

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
            # Process a sample visual observation
            if input_channels is not None:
                # Create a dummy sample with the correct number of channels
                # NOTE: use all zeros as sample because the observation space might not reflect the the stacked frames during initialization
                # Shape: (1, input_channels, height, width)
                if len(visual_space.shape) == 3:  # CHW format
                    height, width = visual_space.shape[1], visual_space.shape[2]
                else:  # HWC format
                    height, width = visual_space.shape[0], visual_space.shape[1]
                
                # Create a correctly shaped tensor filled with zeros
                sample = th.zeros((1, input_channels, height, width), dtype=th.float32)
                print(f"Created dummy sample with shape: {sample.shape}")
            else:
                # Standard case - use the observation space's sample method
                sample = th.as_tensor(visual_space.sample(), dtype=th.float32)
                
                # Convert to NCHW format if needed
                if len(sample.shape) == 3:
                    if sample.shape[0] <= 3:  # Already CHW format
                        sample = sample.unsqueeze(0)  # Add batch dimension
                    else:  # HWC format
                        sample = sample.permute(2, 0, 1).unsqueeze(0)  # Convert and add batch
                
            print(f"Sample shape: {sample.shape}")  # Should be (1, C, H, W)
            n_flatten = self.cnn(sample).shape[1]
            print(f"Flattened features size: {n_flatten}")

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
        # Note: we only convert to tensor once if not already a tensor
        if isinstance(observations["visual"], th.Tensor):
            visual_obs = observations["visual"]
            if visual_obs.dtype != th.float32:
                # Convert in place if possible
                visual_obs = visual_obs.to(dtype=th.float32, non_blocking=True)
        else:
            # Pin memory if using CUDA for faster host-to-device transfer
            if th.cuda.is_available():
                # Reuse buffer if possible to reduce memory allocations
                if not hasattr(self, '_visual_tensor_buffer') or self._visual_tensor_buffer.shape[1:] != observations["visual"].shape:
                    shape = observations["visual"].shape
                    self._visual_tensor_buffer = th.empty((1,) + shape if len(shape) == 3 else shape, dtype=th.float32, device="cuda")
                
                # Copy directly into the buffer
                visual_obs = self._visual_tensor_buffer
                visual_obs.copy_(th.as_tensor(observations["visual"], dtype=th.float32, device="cuda"), non_blocking=True)
            else:
                visual_obs = th.as_tensor(observations["visual"], dtype=th.float32)
        
        # Add batch dimension (CHW -> NCHW)
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        
        visual_features = self.cnn(visual_obs)

        # Process position features
        if isinstance(observations["position"], th.Tensor):
            pos_obs = observations["position"]
            if pos_obs.dtype != th.float32:
                # Convert in place if possible
                pos_obs = pos_obs.to(dtype=th.float32, non_blocking=True)
        else:
            # Use cuda if available for consistency with visual features
            if th.cuda.is_available():
                pos_obs = th.as_tensor(observations["position"], dtype=th.float32, device="cuda")
            else:
                pos_obs = th.as_tensor(observations["position"], dtype=th.float32)
            
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
        # combined = th.cat([visual_features, pos_features], dim=1)
        
        return self.combined(th.cat([visual_features, pos_features], dim=1))


class CustomVecTranspose(VecTransposeImage):
    def __init__(self, venv):
        super().__init__(venv)
        
        # Get the number of stacked frames from the VecFrameStack wrapper
        # Default to 1 if not frame stacked
        self.n_stack = getattr(venv, "n_stack", 1)
        
        # Get the number of channels from the original visual observation space
        # This is the number of channels AFTER stacking (e.g., 12 for 4 stacks of RGB)
        orig_visual_shape = venv.observation_space["visual"].shape
        n_channels = orig_visual_shape[-1]  # Last dimension is channels in HWC format
        
        print(f"Original visual shape: {orig_visual_shape}, detected {n_channels} channels")

        # Update observation space to handle dictionary
        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0,
                    high=255,
                    shape=(n_channels, SCREEN_HEIGHT, SCREEN_WIDTH),  # CHW format with stacked frames
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
        # Fast path for common case to avoid redundant operations
        if isinstance(obs, dict):
            # For single observations (e.g., during evaluation)
            visual_obs = obs["visual"]  # Shape: (H, W, C)
            position = obs["position"]  # Shape: (2,)
        else:
            # For vectorized observations (e.g., from DummyVecEnv)
            # This is the most common case during training
            visual_obs = obs[0]["visual"]  # Shape: (H, W, C) or (1, H, W, C)
            position = obs[0]["position"]  # Shape: (2,) or (1, 2)
            # action_history = obs[0]["action_history"]  # Shape: (10,) or (1, 10)

        # Single fast check for both visual and position to remove batch dim
        # Remove any extra batch dimension from DummyVecEnv if present
        if isinstance(visual_obs, np.ndarray) and len(visual_obs.shape) == 4:
            visual_obs = visual_obs[0]  # Convert (1, H, W, C) to (H, W, C)
        
        if isinstance(position, np.ndarray) and len(position.shape) == 2:
            position = position[0]  # Convert (1, 2) to (2,)

        # Convert from HWC to CHW format - only once and only if needed
        # Check if already in CHW format to avoid unnecessary transpose
        if isinstance(visual_obs, np.ndarray) and len(visual_obs.shape) == 3 and visual_obs.shape[2] <= 12:
            # Standard case - needs transpose

            h, w, c = visual_obs.shape
            
            # Create or resize transpose buffer if needed
            if not hasattr(self, '_transpose_buffer') or self._transpose_buffer.shape != (c, h, w):
                # First usage or shape changed, create new buffer with exact shape needed
                self._transpose_buffer = np.empty((c, h, w), dtype=visual_obs.dtype)
            
            # Manual optimized transpose to reuse existing buffer (no new allocation)
            # it should faster than np.transpose which would create a new array
            for i in range(c):
                # Direct slice assignment is more efficient than np.copyto
                self._transpose_buffer[i] = visual_obs[:, :, i]
            
            visual_result = self._transpose_buffer
        else:
            # Already in the right format or special case, no need to transpose
            visual_result = visual_obs
            
        return {
            "visual": visual_result,
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


# Helper for creating an environment with proper frame stacking
def create_env_with_frame_stacking(n_stack=4):
    """
    Create an environment with frame stacking properly configured for 
    dictionary observations and CHW format.
    
    Args:
        n_stack: Number of frames to stack
        
    Returns:
        A properly configured environment with frame stacking
    """
    # Create base environment
    env = DummyVecEnv([make_env])
    print(f"Original observation space: {env.observation_space}")
    
    # Apply frame stacking
    env = VecFrameStack(env, n_stack=n_stack)
    print(f"After VecFrameStack: {env.observation_space}")
    
    # Apply custom transpose
    env = CustomVecTranspose(env)
    print(f"After CustomVecTranspose: {env.observation_space}")
    
    return env, 3 * n_stack  # Return env and number of input channels


# Helper for creating an evaluation environment with rendering support
def create_eval_env_with_frame_stacking(n_stack=4, render_mode="human", debug=False):
    """
    Create an evaluation environment with frame stacking and rendering support.
    
    Args:
        n_stack: Number of frames to stack (must match training)
        render_mode: The rendering mode to use (e.g., "human")
        debug: Whether to enable debug output for observations
        
    Returns:
        A properly configured environment for evaluation
    """
    # Create a function that returns a properly configured env with render mode
    def make_env_with_render():
        env = VisualSimulationEnv(render_mode=render_mode)
        env = Monitor(env)

        return env
    
    # Create vectorized environment
    env = DummyVecEnv([make_env_with_render])
    print(f"Eval env - original observation space: {env.observation_space}")
    
    # Apply frame stacking - same as in training
    env = VecFrameStack(env, n_stack=n_stack)
    print(f"Eval env - after VecFrameStack: {env.observation_space}")
    
    # Apply custom transpose - same as in training
    env = CustomVecTranspose(env)
    print(f"Eval env - after CustomVecTranspose: {env.observation_space}")
    
    return env


# Custom callback to manage memory during training
class MemoryManagementCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        # Run garbage collection when memory is low or every 50K steps
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 95 or self.num_timesteps % 50000 == 0:
            gc.collect()
            if th.cuda.is_available():
                # Clear CUDA cache if using GPU
                th.cuda.empty_cache()
            
            # Log memory usage if verbose
            if self.verbose > 0:
                import psutil
                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                print(f"Step {self.num_timesteps}: Memory usage: {memory_usage:.2f} MB")
        
        return True


# Only run training code when script is executed directly (not imported)
if __name__ == "__main__":
    # Create environment with frame stacking
    n_stack = 4  # Stack 4 frames
    env, input_channels = create_env_with_frame_stacking(n_stack=n_stack)

    # Configure model parameters
    policy_kwargs = dict(
        features_extractor_class=VisionExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            # Pass the correct number of input channels for the stacked frames
            input_channels=input_channels
        ),
        # Use a larger network to process the stacked frames
        net_arch=[256, 128, 64],
    )

    # Initialize model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        tensorboard_log=os.path.join("Training", "Logs"),
        device=device,
    )

    # Training parameters
    total_timesteps = 1000000
    check_freq = 50000  # Save checkpoint every 50k steps
    MODEL_TAG = "PPO_FrameStack4"

    # Configure garbage collection
    gc.enable()

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
        model = PPO.load(latest_checkpoint, env=env)
        # Extract the timestep from the checkpoint name
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

    # Create memory management callback
    memory_callback = MemoryManagementCallback(verbose=1)

    # Create callback list to combine multiple callbacks
    from stable_baselines3.common.callbacks import CallbackList
    callback_list = CallbackList([checkpoint_callback, memory_callback])

    # Start training with proper checkpointing and memory management
    try:
        if hasattr(th, 'set_num_threads'):
            num_threads = max(8, os.cpu_count())
            th.set_num_threads(num_threads)
            print(f"Setting PyTorch threads to {num_threads}")
        
        # Configure PyTorch to release memory more aggressively
        if hasattr(th.cuda, 'empty_cache'):
            th.cuda.empty_cache()
        
        model.learn(
            total_timesteps=remaining_timesteps,
            progress_bar=True,
            callback=callback_list,
            log_interval=100,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        # Run garbage collection before saving to reduce memory footprint
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()
        
        save_checkpoint(model, start_timestep + model.num_timesteps, model_tag=MODEL_TAG)
        print("Checkpoint saved. You can resume training later.")

    # Clean up memory before saving final model
    gc.collect()
    if th.cuda.is_available():
        th.cuda.empty_cache()

    # Save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"ppo_visual_attention_full_{timestamp}")
    print(f"Final model saved as ppo_visual_attention_full_{timestamp}")
    print("Model training complete!")
    print("To properly evaluate the model with human rendering, use:")
    print(f"python evaluate_model.py ppo_visual_attention_full_{timestamp}")
    print("This will handle the observation stacking and preprocessing correctly for visual rendering.")
