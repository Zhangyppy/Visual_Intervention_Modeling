import pickle
import time
import math
import os
import pygame
import random
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    SubprocVecEnv,
)
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

# Pygame settings
pygame.init()
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
OBS_SIZE = 200  # Observation window size
EVENT_AREA_WIDTH = SCREEN_WIDTH
EVENT_AREA_HEIGHT = SCREEN_HEIGHT
MIN_BLOCK_GENERATION_DISTANCE = 50  # Minimum distance between agent and target for block generation
GRID_SIZE = 10

# Initialize the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Visual Attention Task")
clock = pygame.time.Clock()

# Define colors for rendering
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Constants for game objects
DOT_RADIUS = 10  # Radius of the agent's red dot
BLOCK_SIZE = 60  # Size of the target block
MOVE_SPEED = 5  # Speed of agent's movement


class VisualSimulationEnv(gym.Env):
    """Environment for the visual attention task."""

    def __init__(self, render_mode=None):
        super(VisualSimulationEnv, self).__init__()

        # Metadata for rendering modes
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        # --- ACTION SPACE ---
        # Flattened action space: [x, y] movement and [layer selection]
        self.action_space = spaces.MultiDiscrete([20, 20, 2])

        # --- OBSERVATION SPACE ---
        # Observation space consists of RGB image of size (OBS_SIZE x OBS_SIZE)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(OBS_SIZE, OBS_SIZE, 3),  # height, width, channels
            dtype=np.uint8
        )

        # --- Episode Variables ---
        self._ep_length = 100  # Max episode length
        self._steps = 0  # Current step count
        self.reward = 0  # Reward for the current episode
        self.reward_count = 0  # Counter for reward accumulation

        # Randomly choose block and agent layers
        self.block_layer = np.random.choice(["environment", "notification"])
        self.block_size = BLOCK_SIZE // 2 if self.block_layer == "environment" else BLOCK_SIZE
        self.agent_layer = np.random.choice(["environment", "notification"])

        # Initialize positions of agent and target
        self._initialize_positions()

        # Initialize render settings if required
        if self.render_mode == "human":
            self._init_render()
        else:
            self.screen = None

        self.visited_positions = set()

    def _initialize_positions(self):
        """Initialize positions for agent and target."""
        # Position of the red dot (agent)
        self.dot_pos = np.array(
            [random.randint(DOT_RADIUS, SCREEN_WIDTH - DOT_RADIUS),
             random.randint(DOT_RADIUS, SCREEN_HEIGHT - DOT_RADIUS)],
            dtype=np.float32,
        )

        # Random position for target (block) ensuring minimum distance from agent
        while True:
            self.block_pos = np.array(
                [random.randint(self.block_size // 2, SCREEN_WIDTH - self.block_size // 2),
                 random.randint(self.block_size // 2, SCREEN_HEIGHT - self.block_size // 2)],
                dtype=np.float32,
            )

            if np.linalg.norm(self.dot_pos - self.block_pos) > MIN_BLOCK_GENERATION_DISTANCE:
                break

    def _init_render(self):
        """Initialize the rendering window."""
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def _check_collision(self):
        """Check if the agent collides with the target block."""
        target_center = self.block_pos
        distance = np.linalg.norm(self.dot_pos - target_center)
        return distance < (DOT_RADIUS + self.block_size / 2)

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self._initialize_positions()
        self.reward = 0
        self._steps = 0
        self.reward_count = 0
        self.block_layer = np.random.choice(["environment", "notification"])
        self.block_size = BLOCK_SIZE // 2 if self.block_layer == "environment" else BLOCK_SIZE
        self.agent_layer = np.random.choice(["environment", "notification"])

        return self._get_obs(), {}

    def step(self, action):
        """Advance one step in the environment based on the given action."""
        self._steps += 1
        done = False

        # Action: Move agent and set layer
        self.dot_pos = np.array([action[0] * (SCREEN_WIDTH // 20),
                                 action[1] * (SCREEN_HEIGHT // 20)],
                                dtype=np.float32)
        self.agent_layer = "environment" if action[2] == 0 else "notification"

        # Check if the agent goes out of bounds
        hit_boundary = (self.dot_pos[0] <= DOT_RADIUS or
                        self.dot_pos[0] >= SCREEN_WIDTH - DOT_RADIUS or
                        self.dot_pos[1] <= DOT_RADIUS or
                        self.dot_pos[1] >= SCREEN_HEIGHT - DOT_RADIUS)

        # --- Reward Structure ---
        if self.agent_layer == self.block_layer:
            if self._check_collision():
                self.reward += 200  # Reward for correct collision
                self.reward_count += 1
                if self.reward_count >= 3:
                    done = True  # End episode if target is reached multiple times
        else:
            self.reward -= 10  # Penalty for incorrect layer

        if self._steps >= self._ep_length:
            done = True

        return self._get_obs(), self.reward, done, False, {}

    def _get_obs(self):
        """Generate the current observation (image) for the agent."""
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(WHITE)

        # Draw blue block (target)
        pygame.draw.rect(canvas, BLUE,
                         (int(self.block_pos[0] - self.block_size // 2),
                          int(self.block_pos[1] - self.block_size // 2),
                          self.block_size, self.block_size))

        # Draw red dot (agent)
        pygame.draw.circle(canvas, RED, self.dot_pos.astype(int), DOT_RADIUS)

        # Draw boundary (observation box)
        pygame.draw.rect(canvas, BLACK, (0, 0, EVENT_AREA_WIDTH, EVENT_AREA_HEIGHT), 3)

        # Convert canvas to numpy array for observation
        obs = pygame.surfarray.array3d(canvas)  # Convert to (W, H, C)
        obs = np.transpose(obs, (1, 0, 2))  # Convert to (H, W, C)
        return obs

    def render(self, mode='human'):
        """Render the environment for visual inspection."""
        if self.render_mode is None:
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(WHITE)

        # Draw the blue block and red dot (within the observation box)
        pygame.draw.rect(canvas, BLUE,
                         (int(self.block_pos[0] - self.block_size // 2),
                          int(self.block_pos[1] - self.block_size // 2),
                          self.block_size, self.block_size))
        pygame.draw.circle(canvas, RED, self.dot_pos.astype(int), DOT_RADIUS)

        # Display additional info (reward, steps, layer)
        font = pygame.font.SysFont("Arial", 10)

        reward_text = font.render(f"Reward: {self.reward}", True, BLACK)
        canvas.blit(reward_text, (10, 10))

        step_text = font.render(f"Steps: {self._steps}", True, BLACK)
        canvas.blit(step_text, (10, 20))

        layer_text = font.render(f"Layer: {self.agent_layer}", True, BLACK)
        canvas.blit(layer_text, (10, 30))

        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array mode
            return np.transpose(pygame.surfarray.array3d(canvas), axes=(1, 0, 2))

    def close(self):
        """Close the environment and the Pygame window."""
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    """Feature extractor for the environment's visual inputs."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Convolutional layers for image feature extraction
        n_input_channels = observation_space.shape[0]
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, 5, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 128, 3, stride=2),
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Calculate flattened feature size
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            th.nn.LayerNorm(features_dim),
            th.nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.cnn(observations)
        return self.linear(features)


# Initialize environment
env = VisualSimulationEnv()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

# Model configuration
policy_kwargs = dict(
    features_extractor_class=VisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 64]
)

# Initialize PPO model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    tensorboard_log=os.path.join('Training', 'Logs')
)


# # 开始训练
# model.learn(total_timesteps=5000000)
#
# # 保存模型
# model.save("ppo_visual_attention_v4")


# # Load pre-trained model
model = PPO.load("../find_notification_based_vision/ppo_visual_attention_v4.zip")

# Create evaluation environment
eval_env = VisualSimulationEnv(render_mode='human')
eval_env = Monitor(eval_env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# To view TensorBoard logs:
# tensorboard --logdir=find_notification_based_vision/Training/Logs
