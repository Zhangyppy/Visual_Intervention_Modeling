#import gym
#from gym import Env
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

# Pygame settings
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
OBS_SIZE = 100  # Observation window size

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Visual Attention Task")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Constants
DOT_RADIUS = 10
BLOCK_SIZE = 90
MOVE_SPEED = 20
# GREEN_BOX_SIZE = 100  # Green box is slightly bigger than the blue block

class VisualSimulationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(VisualSimulationEnv, self).__init__()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        # --- ACTION SPACE ---
        # We will choose from 4 discrete directions:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # Observation space: image format (height, width, channels)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(OBS_SIZE, OBS_SIZE, 3),
            dtype=np.uint8
        )

        # Episode variables
        self._ep_length = 50  # Max episode length
        self._steps = 0
        self.reward = 0

        # Initialize state
        self.dot_pos = [
            random.randint(DOT_RADIUS, SCREEN_WIDTH-DOT_RADIUS),
            random.randint(DOT_RADIUS, SCREEN_HEIGHT-DOT_RADIUS)
        ]
        self.block_pos = np.array([
            random.randint(BLOCK_SIZE, SCREEN_WIDTH-BLOCK_SIZE),
            random.randint(BLOCK_SIZE, SCREEN_HEIGHT-BLOCK_SIZE)
        ])
        self.green_box_pos = self.dot_pos  # Green box follows red dot's position

        # Initialize pygame for rendering
        if self.render_mode == "human":
            self._init_render()
        else:
            self.screen = None

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        # Initialize or reset the environment state
        super().reset(seed=seed)
        self.dot_pos = [
            random.randint(DOT_RADIUS, SCREEN_WIDTH-DOT_RADIUS),
            random.randint(DOT_RADIUS, SCREEN_HEIGHT-DOT_RADIUS)
        ]
        self.block_pos = np.array([
            random.randint(BLOCK_SIZE, SCREEN_WIDTH-BLOCK_SIZE),
            random.randint(BLOCK_SIZE, SCREEN_HEIGHT-BLOCK_SIZE)
        ])
        self.green_box_pos = self.dot_pos

        self.reward = 0
        self._steps = 0

        return self._get_obs(), {}

    def step(self, action):

        self._steps += 1
        done = False

        # Move the red dot based on action
        if action == 0:  # Up
            self.dot_pos[1] = max(0, self.dot_pos[1] - MOVE_SPEED)
        elif action == 1:  # Right
            self.dot_pos[0] = min(SCREEN_WIDTH, self.dot_pos[0] + MOVE_SPEED)
        elif action == 2:  # Down
            self.dot_pos[1] = min(SCREEN_HEIGHT, self.dot_pos[1] + MOVE_SPEED)
        elif action == 3:  # Left
            self.dot_pos[0] = max(0, self.dot_pos[0] - MOVE_SPEED)

        # Green box follows red dot's position
        self.green_box_pos = self.dot_pos

        # Green box around red dot
        green_box_rect = pygame.Rect(
            self.green_box_pos[0] - OBS_SIZE // 2,
            self.green_box_pos[1] - OBS_SIZE // 2,
            OBS_SIZE ,
            OBS_SIZE ,
        )

        block_rect = pygame.Rect(
            self.block_pos[0] - BLOCK_SIZE // 2,
            self.block_pos[1] - BLOCK_SIZE // 2,
            BLOCK_SIZE,
            BLOCK_SIZE,
        )

        dot_rect = pygame.Rect(
            self.dot_pos[0] - DOT_RADIUS,
            self.dot_pos[1] - DOT_RADIUS,
            DOT_RADIUS * 2,
            DOT_RADIUS * 2,
        )

        # # Calculate reward based on block position and intersection with green box
        # observation_area = OBS_SIZE  * OBS_SIZE
        # intersection_area = green_box_rect.clip(block_rect).width * green_box_rect.clip(block_rect).height
        # reward_for_intersection = intersection_area / observation_area * 10  # Reward based on overlap
        #
        # self.reward = -0.1 + reward_for_intersection

        if self.is_blue_square_in_green_box():
            self.reward += 100
            done = True

        info = {}
        truncated = False

        return self._get_obs(), self.reward, done, truncated, info

    def _get_obs(self):

        # Extract the observation window (centered around the red dot and green box)
        x_min = max(0, self.dot_pos[0] - OBS_SIZE // 2)
        y_min = max(0, self.dot_pos[1] - OBS_SIZE // 2)
        x_max = min(SCREEN_WIDTH, self.dot_pos[0] + OBS_SIZE // 2)
        y_max = min(SCREEN_HEIGHT, self.dot_pos[1] + OBS_SIZE // 2)

        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        sub_surface = canvas.subsurface((x_min, y_min, x_max - x_min, y_max - y_min))
        obs = pygame.surfarray.array3d(sub_surface)

        final_obs = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
        final_obs[:obs.shape[0], :obs.shape[1]] = obs

        return final_obs

    def render(self, mode='rgb_array'):
        if self.render_mode is None:
            return

            # 创建画布（如果不存在）
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

            # 绘制元素
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(WHITE)

        # Draw the blue block and red dot (within the green observation box)
        pygame.draw.rect(canvas, BLUE, (*self.block_pos - BLOCK_SIZE // 2, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.circle(canvas, RED, np.array(self.dot_pos).astype(int), DOT_RADIUS)

        # Define the green observation box
        x, y = self.green_box_pos
        pygame.draw.rect(canvas, GREEN,
                         (x - OBS_SIZE // 2, y - OBS_SIZE // 2, OBS_SIZE, OBS_SIZE), 2)

        # 显示奖励值
        font = pygame.font.SysFont("Arial", 24)
        text = font.render(f"Reward: {self.reward}", True, BLACK)
        canvas.blit(text, (10, 10))

        # 根据渲染模式处理显示
        if self.render_mode == "human":
            self.screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array模式
            return np.transpose(
                pygame.surfarray.array3d(canvas),
                axes=(1, 0, 2)
            )

        # if mode == 'rgb_array':
        #     return self._get_obs()
        # elif mode == 'human':
        #     if self.screen is None:
        #         pygame.init()
        #         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        #     self.screen.blit(pygame.surfarray.make_surface(self._get_obs()), (0, 0))
        #     pygame.display.flip()
        #     self.clock.tick(30)

    def is_blue_square_in_green_box(self):
        # 获取绿色方框的边界
        green_box_left = self.green_box_pos[0] - OBS_SIZE // 2
        green_box_right = self.green_box_pos[0] + OBS_SIZE // 2
        green_box_top = self.green_box_pos[1] - OBS_SIZE // 2
        green_box_bottom = self.green_box_pos[1] + OBS_SIZE // 2

        # 获取蓝色方块的边界
        blue_square_left = self.block_pos[0] - BLOCK_SIZE // 2
        blue_square_right = self.block_pos[0] + BLOCK_SIZE // 2
        blue_square_top = self.block_pos[1] - BLOCK_SIZE // 2
        blue_square_bottom = self.block_pos[1] + BLOCK_SIZE // 2

        # 判断蓝色方块是否完全在绿色方框内
        return (green_box_left <= blue_square_left and
                green_box_right >= blue_square_right and
                green_box_top <= blue_square_top and
                green_box_bottom >= blue_square_bottom)

    def close(self):
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    """Feature extractor for the visual environment input."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(3, 32, 5, stride=2),  # 100x100 -> 48x48
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 3, stride=2),  # 48x48 -> 23x23
            th.nn.ReLU(),
            th.nn.Conv2d(64, 128, 3, stride=2),  # 23x23 -> 10x10
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Compute the number of flattened units
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            th.nn.LayerNorm(features_dim),
            th.nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = observations.permute(0, 1, 3, 2)  # Convert to (batch_size, channels, height, width)
        return self.linear(self.cnn(observations))


# Create environment pipeline and wrap the evaluation environment with Monitor
env = VisualSimulationEnv()
env = Monitor(env)

# Create environment
env = DummyVecEnv([lambda: env])  # This creates a vectorized environment
env = VecTransposeImage(env)  # Ensure the observation space is in the right format


# Define policy kwargs
policy_kwargs = dict(
    features_extractor_class=VisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 64]  # Modify the architecture as needed
)

# Initialize PPO model with the custom policy
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_visual_attention_tensorboard/"
)

# Then, pass the wrapped environment to the model
model.learn(total_timesteps=10000)

# Save the model after training
model.save("ppo_visual_attention")

# model = PPO.load("../find_notification_based_vision/ppo_visual_attention.zip")

# 创建评估环境
eval_env = VisualSimulationEnv(render_mode='human')
eval_env = Monitor(eval_env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# # Example of testing the model
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, reward, done, truncated, info = env.step(action)
#     env.render(mode="human")
#
# env.close()
