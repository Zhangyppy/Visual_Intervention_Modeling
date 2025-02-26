#import gym
#from gym import Env
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import random
import pickle
import time

from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
import pygame
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


# Pygame settings
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Visual Attention Task")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Constants
DOT_RADIUS = 10
BLOCK_SIZE = 90
MOVE_SPEED = 20

class VisualSimulationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(VisualSimulationEnv, self).__init__()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        # --- ACTION SPACE ---
        # We will choose from 4 discrete directions:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # --- OBSERVATION SPACE ---
        # For simplicity, let's keep track of:
        # [dot_x, dot_y, block_x, block_y]
        # If you want to observe the line direction too, you can extend this to shape=(6,)
        self.observation_space = spaces.Box(
            low=0, high=max(screen_width, screen_height), shape=(4,), dtype=np.float32
        )

        # --- Episode Variables ---
        self._ep_length = 50  # Max episode length
        self._steps = 0  # Current step

        # Initialize state
        self.dot_pos = [
            random.randint(0, screen_width),
            random.randint(0, screen_height)
        ]
        self.block_pos = [
            random.randint(0, screen_width),
            random.randint(0, screen_height)
        ]

        #self.current_block_index = 0
        self.reward = 0

        # Count the number of reached notification
        self.count = 0

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.dot_pos = [
            random.randint(0, screen_width),
            random.randint(0, screen_height)
        ]

        self.block_pos = [
            random.randint(0, screen_width),
            random.randint(0, screen_height)
        ]

        self.reward = 0
        self._steps = 0
        self.count = 0

        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1

        done = False

        # Move the red dot based on action
        if action == 0:  # Up
            self.dot_pos[1] = max(0, self.dot_pos[1] - MOVE_SPEED)
        elif action == 1:  # Right
            self.dot_pos[0] = min(screen_width, self.dot_pos[0] + MOVE_SPEED)
        elif action == 2:  # Down
            self.dot_pos[1] = min(screen_height, self.dot_pos[1] + MOVE_SPEED)
        elif action == 3:  # Left
            self.dot_pos[0] = max(0, self.dot_pos[0] - MOVE_SPEED)

        # Check if the dot is within the block
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

        # Calculate distance to the goal (normalized reward)
        distance_to_goal = np.linalg.norm(np.array(self.dot_pos) - np.array(self.block_pos))
        max_distance = np.linalg.norm(np.array([screen_width, screen_height]))

        # Reward shaping: give a higher reward for being closer to the block
        self.reward = -0.1 + (max_distance - distance_to_goal) / max_distance

        # Increase reward when reaching the goal
        if block_rect.colliderect(dot_rect):
            self.reward += 100
            done = True

        # NOTE early stopping if the episode length exceeds the limit
        # if self._steps >= self._ep_length:
        #     done = True

        # Define a success threshold
        #done = self.count > 10
        truncated = False
        info = {}

        return self._get_obs(), self.reward, done, truncated, info

    def _get_obs(self):
        # Return observation
        return np.array(
            [
                self.dot_pos[0],
                self.dot_pos[1],
                self.block_pos[0],
                self.block_pos[1],
            ],
            dtype=np.float32,
        )

    def render(self, mode='rgb_array'):
        if self.render_mode is None:
            return

            # 创建画布（如果不存在）
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))

            # 绘制元素
        canvas = pygame.Surface((screen_width, screen_height))
        canvas.fill(WHITE)

        # 绘制红点
        pygame.draw.circle(canvas, RED, self.dot_pos, DOT_RADIUS)

        # 绘制蓝块
        pygame.draw.rect(
            canvas,
            BLUE,
            (
                self.block_pos[0] - BLOCK_SIZE // 2,
                self.block_pos[1] - BLOCK_SIZE // 2,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )

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

    def close(self):
        if self.screen is not None:
            pygame.quit()


train_env = VisualSimulationEnv()  # 训练时不需要渲染
train_env = Monitor(train_env)

# 创建模型
log_path = os.path.join('Training', 'Logs')
model = PPO("MlpPolicy", train_env, verbose=1,  tensorboard_log=log_path)

# 训练模型run
model.learn(total_timesteps=50000)

#model = PPO.load("PPO.zip")

# 创建评估环境
eval_env = VisualSimulationEnv(render_mode='human')
eval_env = Monitor(eval_env)

# 评估模型
mean_reward, std_reward = evaluate_policy(
    model,
    eval_env,
    n_eval_episodes=10,
    render=True
)

# Save model
model.save('PPO')
evaluate_policy(model, train_env, n_eval_episodes=10, render=True)

