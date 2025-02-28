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
OBS_SIZE = 200  # 观测窗口大小

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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
        # --- ACTION SPACE ---
        # We will choose from 4 discrete directions:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # --- OBSERVATION SPACE ---
        # For simplicity, let's keep track of:
        # [dot_x, dot_y, block_x, block_y]
        # If you want to observe the line direction too, you can extend this to shape=(6,)
        # 修改观测空间为图像格式 (H, W, C)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(OBS_SIZE, OBS_SIZE, 3),  # 确保是(height, width, channels)
            dtype=np.uint8
        )

        # --- Episode Variables ---
        self._ep_length = 50  # Max episode length
        self._steps = 0  # Current step

        # Initialize state as numpy arrays
        # note: is always good to explicitly specify the type if you can
        # thus we can reduce the type inference error  
        # since the interpreter may infer the wrong type
        self.dot_pos = np.array([
            random.randint(DOT_RADIUS, SCREEN_WIDTH-DOT_RADIUS),
            random.randint(DOT_RADIUS, SCREEN_HEIGHT-DOT_RADIUS)
        ], dtype=np.float32)
        
        self.block_pos = np.array([
            random.randint(BLOCK_SIZE, SCREEN_WIDTH-BLOCK_SIZE),
            random.randint(BLOCK_SIZE, SCREEN_HEIGHT-BLOCK_SIZE)
        ], dtype=np.float32)

        # Initialize pygame
        if self.render_mode == "human":
            self._init_render()
        else:
            self.screen = None

        #self.current_block_index = 0
        self.reward = 0
        
    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.dot_pos = np.array([
            random.randint(DOT_RADIUS, SCREEN_WIDTH-DOT_RADIUS),
            random.randint(DOT_RADIUS, SCREEN_HEIGHT-DOT_RADIUS)
        ], dtype=np.float32)
        
        self.block_pos = np.array([
            random.randint(BLOCK_SIZE, SCREEN_WIDTH-BLOCK_SIZE),
            random.randint(BLOCK_SIZE, SCREEN_HEIGHT-BLOCK_SIZE)
        ], dtype=np.float32)

        self.reward = 0
        self._steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self._steps += 1

        done = False

        # 移动红点
        action_map = {
            0: np.array([0, -MOVE_SPEED]),  # 上
            1: np.array([MOVE_SPEED, 0]),   # 右
            2: np.array([0, MOVE_SPEED]),   # 下
            3: np.array([-MOVE_SPEED, 0])   # 左
        }

        self.dot_pos = np.clip(
            self.dot_pos + action_map[action],
            [0, 0],
            [SCREEN_WIDTH, SCREEN_HEIGHT]
        )

        # 计算奖励
        distance = np.linalg.norm(self.dot_pos - self.block_pos)
        max_dist = np.sqrt(SCREEN_WIDTH ** 2 + SCREEN_HEIGHT ** 2)
        reward = -0.1 + (max_dist - distance) / max_dist

        # 碰撞检测
        if np.linalg.norm(self.dot_pos - self.block_pos) < BLOCK_SIZE / 2 + DOT_RADIUS:
            reward += 100
            done = True
        else:
            done = False

        # Check if episode is done due to step limit
        # note: double check if you need this
        # I saw you declare _ep_length but didn't use it
        if self._steps >= self._ep_length:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # 生成观测窗口
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill((255, 255, 255))

        # 绘制蓝块
        pygame.draw.rect(
            canvas, (0, 0, 255),
            (
                int(self.block_pos[0] - BLOCK_SIZE // 2),
                int(self.block_pos[1] - BLOCK_SIZE // 2),
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )

        # 绘制红点
        pygame.draw.circle(
            canvas, (255, 0, 0),
            self.dot_pos.astype(int), DOT_RADIUS
        )

        # 截取观察区域（以红点为中心）
        x_min = max(0, int(self.dot_pos[0] - OBS_SIZE // 2))
        y_min = max(0, int(self.dot_pos[1] - OBS_SIZE // 2))
        
        # calculate the width and height of the observation
        # thus we can check if the observation is out of bounds
        width = min(OBS_SIZE, SCREEN_WIDTH - x_min)
        height = min(OBS_SIZE, SCREEN_HEIGHT - y_min)
        
        if width <= 0 or height <= 0:
            # Handle edge case - return a black window in HWC format
            return np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
            
        # another edge case - ensure the dimensions are valid
        try:
            sub_surface = canvas.subsurface((x_min, y_min, width, height))
            obs = pygame.surfarray.array3d(sub_surface)
            
            # PyGame returns in (W, H, C) format, convert to (H, W, C)
            obs = obs.transpose(1, 0, 2)
            
            # 填充不足区域，确保大小为 (OBS_SIZE, OBS_SIZE, 3)
            final_obs = np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)
            final_obs[:min(height, OBS_SIZE), :min(width, OBS_SIZE)] = obs[:min(height, OBS_SIZE), :min(width, OBS_SIZE)]
            
            # Return in HWC format (Height, Width, Channel)
            # VecTransposeImage will convert to CHW format for PyTorch
            # no need to permute it here
            return final_obs
        
        except ValueError as e:
            print(f"Error creating observation: {e}")
            print(f"Dimensions: x_min={x_min}, y_min={y_min}, width={width}, height={height}")
            # Return an empty observation in HWC format
            return np.zeros((OBS_SIZE, OBS_SIZE, 3), dtype=np.uint8)

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self._get_obs()
        elif mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                self.clock = pygame.time.Clock()
            
            # Get observation in HWC format
            obs = self._get_obs()
            
            # For display, PyGame expects (W, H, C)
            # so we need to convert it to (H, W, C) for rendering
            obs_for_display = obs.transpose(1, 0, 2)
            
            try:
                surf = pygame.surfarray.make_surface(obs_for_display)
                self.screen.fill(WHITE)
                self.screen.blit(surf, (0, 0))
                pygame.display.flip()
                self.clock.tick(30)
            except Exception as e:
                print(f"Error in render: {e}")
                print(f"Observation shape: {obs_for_display.shape}")

    def close(self):
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    """适应200x200输入的特征提取器"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # After VecTransposeImage wrapper, images are already in [C, H, W] format
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


# 创建环境管道
def make_env():
    env = VisualSimulationEnv()
    env = Monitor(env)
    return env

env = DummyVecEnv([lambda: VisualSimulationEnv()])
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
model.learn(total_timesteps=100000)

# 评估模型
eval_env = VisualSimulationEnv(render_mode='human')
mean_reward, _ = evaluate_policy(model, Monitor(eval_env), n_eval_episodes=10, render=True)

# 保存模型
model.save("ppo_visual_attention_v2")



