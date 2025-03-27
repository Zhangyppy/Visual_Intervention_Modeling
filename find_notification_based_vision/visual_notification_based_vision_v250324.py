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
import math
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
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
OBS_SIZE = 200  # 观测窗口大小
EVENT_AREA_WIDTH = SCREEN_WIDTH
EVENT_AREA_HEIGHT = SCREEN_HEIGHT
MIN_BLOCK_GENERATION_DISTANCE = (
    50  # Minimum distance between agent and target to generate a block
)
GRID_SIZE = 10

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Visual Attention Task")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)

# Constants
DOT_RADIUS = 10
BLOCK_SIZE = 60
MOVE_SPEED = 5


class VisualSimulationEnv(gym.Env):
    def __init__(self, render_mode=None):

        super(VisualSimulationEnv, self).__init__()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        # --- ACTION SPACE ---
        # Flattened action space: 2D move action and 1D layer action
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        # --- OBSERVATION SPACE ---
        # For simplicity, let's keep track of:
        # [dot_x, dot_y, block_x, block_y]
        # If you want to observe the line direction too, you can extend this to shape=(6,)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(OBS_SIZE, OBS_SIZE, 3),  # 确保是(height, width, channels)
            dtype=np.uint8
        )

        # --- Episode Variables ---
        self._ep_length = 100  # Max episode length
        self._steps = 0  # Current step

        self.block_layer = np.random.choice(["environment", "notification"])  # 随机选择层
        self.block_size = BLOCK_SIZE // 2 if self.block_layer == "environment" else BLOCK_SIZE
        self.agent_layer = np.random.choice(["environment", "notification"])

        self._initialize_positions()
        self.visited_positions = set()

        # Initialize pygame
        if self.render_mode == "human":
            self._init_render()
        else:
            self.screen = None



        # self.current_block_index = 0
        self.reward = 0
        self.reward_count = 0

    def _initialize_positions(self):
        """Initialize agent and target positions"""

        self.dot_pos = np.array(
            [
                random.randint(DOT_RADIUS, SCREEN_WIDTH - DOT_RADIUS),
                random.randint(DOT_RADIUS, SCREEN_HEIGHT - DOT_RADIUS),
            ],
            dtype=np.float32,
        )

        # Place the target at a random position, but ensure it's not right on top of the agent
        while True:
            self.block_pos = np.array(
                [
                    random.randint(self.block_size // 2, SCREEN_WIDTH - self.block_size // 2),
                    random.randint(self.block_size // 2, SCREEN_HEIGHT - self.block_size // 2),
                ],
                dtype=np.float32,
            )

            # Make sure the target is not too close to the agent at initialization
            # NOTE: beware if we scale down the screen size, this value needs to be scaled down as well
            if (
                    np.linalg.norm(self.dot_pos - self.block_pos)
                    > MIN_BLOCK_GENERATION_DISTANCE
            ):
                break

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def _check_collision(self):
        """Check if the agent collides with the target"""
        # Calculate the center of the target
        target_center = self.block_pos

        # Check if the distance between the agent's center and the target's center
        # is less than the sum of the agent's radius and half the target size
        distance = np.linalg.norm(self.dot_pos - target_center)
        return distance < (DOT_RADIUS + self.block_size / 2)

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self._initialize_positions()
        self.visited_positions = set()

        # self.is_environment_layer = random.choice([True, False])

        self.reward = 0
        self._steps = 0
        self.reward_count = 0

        self.block_layer = np.random.choice(["environment", "notification"])  # 随机选择层
        self.block_size = BLOCK_SIZE // 2 if self.block_layer == "environment" else BLOCK_SIZE
        self.agent_layer = np.random.choice(["environment", "notification"])

        # print("Block_Layer : " + self.block_layer)

        return self._get_obs(), {}

    def step(self, action):

        self._steps += 1
        done = False

        movement_action = action[:2]  # x, y 方向
        layer_choice = int(round(action[2]))  # 0: 环境层, 1: 通知层


        self.agent_layer = "environment" if layer_choice == 0 else "notification"
        # print("Agent_Layer : " + self.agent_layer)

        #renew the dot's position
        self.dot_pos = np.array([
            movement_action[0] * SCREEN_WIDTH,
            movement_action[1] * SCREEN_HEIGHT
        ], dtype=np.float32)

        # Check if the intended position would be outside the screen boundaries
        hit_boundary = (
            self.dot_pos[0] <= DOT_RADIUS
            or self.dot_pos[0] >= SCREEN_WIDTH - DOT_RADIUS
            or self.dot_pos[1] <= DOT_RADIUS
            or self.dot_pos[1] >= SCREEN_HEIGHT - DOT_RADIUS
        )

        # --- Reward Structure ---
        # self.reward -= 1

        # if self.agent_layer == self.block_layer:
        #     self.reward += 2
        #     # Terminal reward for reaching target
        #     if self._check_collision():
        #         self.reward += 200  # Big reward for reaching target
        #         self.reward_count += 1
        #         if self.reward_count >= 3:
        #             done = True
        # else:
        #     self.reward -= 5

        if self.agent_layer == self.block_layer:
            self.reward += 200
            self.reward_count += 1
            # print(self.reward)
            if self.reward_count >= 3:
                done = True
        else:
            self.reward -= 5

        # Boundary penalty
        if hit_boundary:
            self.reward -= 10

        # Reward for exploring new positions, use discretized position for our convience
        # Discretize position to track visited areas
        # grid_x = int(self.dot_pos[0] // GRID_SIZE)
        # grid_y = int(self.dot_pos[1] // GRID_SIZE)
        # grid_pos = (grid_x, grid_y)

        # if grid_pos not in self.visited_positions:
        #     self.visited_positions.add(grid_pos)
        #     # diminishing reward for finding new positions
        #     self.reward += 0.5 * math.exp(len(self.visited_positions) * -0.05)

        # Check if episode is done due to step limit
        # note: double check if you need this
        # I saw you declare _ep_length but didn't use it
        if self._steps >= self._ep_length:
            done = True

        return self._get_obs(), self.reward, done, False, {}


    def _get_obs(self):
        # initiate the render window
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill((255, 255, 255))

        # Draw blue_block
        pygame.draw.rect(
            canvas, (0, 0, 255),
            (
                int(self.block_pos[0] - self.block_size // 2),
                int(self.block_pos[1] - self.block_size // 2),
                self.block_size,
                self.block_size
            )
        )

        # Draw red_dot
        pygame.draw.circle(
            canvas,
            RED,
            self.dot_pos.astype(int),
            DOT_RADIUS
        )

        # Draw layers
        # border_color = self.environment_layer_color
        pygame.draw.rect(
            canvas,
            BLACK,
            (0, 0, EVENT_AREA_WIDTH, EVENT_AREA_HEIGHT),
            3
        )

        # 转换为 numpy 数组
        obs = pygame.surfarray.array3d(canvas)  # 变成 (W, H, C)
        obs = np.transpose(obs, (1, 0, 2))  # 变成 (H, W, C)
        return obs


    def render(self, mode='human'):

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
        pygame.draw.rect(
            canvas,
            BLUE,
            (
                int(self.block_pos[0] - self.block_size // 2),
                int(self.block_pos[1] - self.block_size // 2),
                self.block_size,
                self.block_size
            )
        )

        # 绘制红点
        pygame.draw.circle(
            canvas,
            RED,
            self.dot_pos.astype(int),
            DOT_RADIUS
        )

        # Draw layers
        # border_color = self.environment_layer_color
        pygame.draw.rect(
            canvas,
            BLACK,
            (0, 0, EVENT_AREA_WIDTH, EVENT_AREA_HEIGHT),
            3
        )

        # 显示奖励值
        font = pygame.font.SysFont("Arial", 10)
        reward_text = font.render(f"Reward: {self.reward}", True, BLACK)
        canvas.blit(reward_text, (10, 10))

        step_text = font.render(f"Steps: {self._steps}", True, BLACK)
        canvas.blit(step_text, (10, 20))

        layer_text = font.render(f"Layer: {self.agent_layer}", True, BLACK)
        canvas.blit(layer_text, (10, 30))

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

    def close(self):
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    """适应200x200输入的特征提取器"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # After VecTransposeImage wrapper, images are already from HWC to [C, H, W] format
        # So input channels are the first dimension
        n_input_channels = observation_space.shape[0]

        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, 5, stride=2),  # 200x200 -> 98x98
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 3, stride=2),  # 98x98 -> 48x48
            th.nn.ReLU(),
            th.nn.Conv2d(64, 128, 3, stride=2),  # 48x48 -> 23x23
            th.nn.ReLU(),
            th.nn.Flatten(),
        )

        # Calculate the size of the flattened features
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


env = VisualSimulationEnv()
env = Monitor(env)
env = DummyVecEnv([lambda: env])
# use VecTransposeImage wrapper - it will convert from HWC to CHW format
# note: if you want to use libraries, make sure you check what they did
# otherwise you code may very likely to conflict with the library
env = VecTransposeImage(env)

# 配置模型参数
policy_kwargs = dict(
    features_extractor_class=VisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 64]
)

#
# 初始化模型
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

# 开始训练
model.learn(total_timesteps=10000)

# 保存模型
model.save("ppo_visual_attention_v4")

# model = PPO.load("../find_notification_based_vision/ppo_visual_attention_v4.zip")

# 创建评估环境
eval_env = VisualSimulationEnv(render_mode='human')
eval_env = Monitor(eval_env)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=True)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")



# tensorboard --logdir=find_notification_based_vision/Training/Logs