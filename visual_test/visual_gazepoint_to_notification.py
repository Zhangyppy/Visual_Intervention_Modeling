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
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Constants
AGENT_RADIUS = 10
BLOCK_SIZE = 50
MOVE_SPEED = 15

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
        # The observation is the entire screen as an RGB image
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(SCREEN_HEIGHT, SCREEN_WIDTH, 3),  # Full screen (height, width, channels)
            dtype=np.uint8
        )

        # --- Episode Variables ---
        self._ep_length = 200  # Max episode length
        self._steps = 0  # Current step

        # Initialize positions of agent and target
        self._initialize_positions()

        # Initialize pygame if in human render mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            self._init_render()

        self.reward = 0
        self.last_action = None
        
    def _initialize_positions(self):
        """Initialize agent and target positions"""
        self.agent_pos = np.array([
            random.randint(AGENT_RADIUS, SCREEN_WIDTH-AGENT_RADIUS),
            random.randint(AGENT_RADIUS, SCREEN_HEIGHT-AGENT_RADIUS)
        ], dtype=np.float32)
        
        # Place the target at a random position, but ensure it's not right on top of the agent
        while True:
            self.target_pos = np.array([
                random.randint(BLOCK_SIZE//2, SCREEN_WIDTH-BLOCK_SIZE//2),
                random.randint(BLOCK_SIZE//2, SCREEN_HEIGHT-BLOCK_SIZE//2)
            ], dtype=np.float32)
            
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
    
    def step(self, action):
        # Process events to prevent pygame from becoming unresponsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        self._steps += 1
        done = False

        # Move the agent
        action_map = {
            0: np.array([0, -MOVE_SPEED]),  # Up
            1: np.array([MOVE_SPEED, 0]),   # Right
            2: np.array([0, MOVE_SPEED]),   # Down
            3: np.array([-MOVE_SPEED, 0])   # Left
        }

        # Store the previous position
        prev_pos = self.agent_pos.copy()
        
        # Calculate the intended (unclipped) position
        intended_pos = self.agent_pos + action_map[action]
        
        # Check if the intended position would be outside the screen boundaries
        hit_boundary = (
            intended_pos[0] <= AGENT_RADIUS or
            intended_pos[0] >= SCREEN_WIDTH - AGENT_RADIUS or
            intended_pos[1] <= AGENT_RADIUS or
            intended_pos[1] >= SCREEN_HEIGHT - AGENT_RADIUS
        )
        
        # Clip the agent position to the screen boundaries
        self.agent_pos = np.clip(
            intended_pos,
            [AGENT_RADIUS, AGENT_RADIUS],
            [SCREEN_WIDTH - AGENT_RADIUS, SCREEN_HEIGHT - AGENT_RADIUS]
        )

        # Calculate movement distance
        move_distance = np.linalg.norm(self.agent_pos - prev_pos)
        
        # Initialize reward
        reward = -0.1  # Small penalty for each step
        
        # --- Reward Structure ---
        
        # # Calculate distance to target
        # distance = np.linalg.norm(self.agent_pos - self.target_pos)
        # prev_distance = np.linalg.norm(prev_pos - self.target_pos)
        
        # # Reward for getting closer to target
        # if distance < prev_distance:
        #     reward += 0.2
        
        # Boundary penalty
        if hit_boundary:
            reward -= 0.5
        
        # Terminal reward for reaching target
        if self._check_collision():
            reward += 200  # Big reward for reaching target
            done = True
        
        # # Reward for exploring (changing actions)
        # if self.last_action != action:
        #     reward += 0.05
        
        # Check if episode is done due to step limit
        if self._steps >= self._ep_length:
            done = True

        # Store the reward and last action value for displaying in render
        self.reward += reward             
        self.last_action = action

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create a canvas for the full screen observation
        canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        canvas.fill(WHITE)

        # Draw the blue target block
        target_x_min, target_y_min, _, _ = self._get_target_boundaries()
        pygame.draw.rect(
            canvas, BLUE,
            (
                target_x_min,
                target_y_min,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )

        # Draw the red agent
        pygame.draw.circle(
            canvas, RED,
            self.agent_pos.astype(int), AGENT_RADIUS
        )


        # draw the space boundaries outer boundary
        pygame.draw.rect(
            canvas, BLACK,
            (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
            2
        )

        # Convert the canvas to a numpy array
        obs = pygame.surfarray.array3d(canvas)
        # PyGame returns in (W, H, C) format, convert to (H, W, C)
        obs = obs.transpose(1, 0, 2)
        
        return obs

    def render(self, mode='human'):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
            
        if mode == 'rgb_array':
            # For rgb_array mode, return the full screen view
            return self._render_frame()
            
        elif mode == 'human':
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
            self.screen, BLUE,
            (
                target_x_min,
                target_y_min,
                BLOCK_SIZE,
                BLOCK_SIZE
            )
        )
        
        # Draw the red agent
        pygame.draw.circle(
            self.screen, RED,
            self.agent_pos.astype(int), AGENT_RADIUS
        )

        # draw the space boundaries outer boundary
        pygame.draw.rect(
            self.screen, BLACK,
            (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
            2
        )
        
        # Draw text info
        font = pygame.font.SysFont('Arial', 14)
        
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

        # Display last action
        action_names = {0: "Up", 1: "Right", 2: "Down", 3: "Left", None: "None"}
        last_action_name = action_names.get(self.last_action, str(self.last_action))
        last_action_text = font.render(f"Last Action: {last_action_name}", True, BLACK)
        self.screen.blit(last_action_text, (10, 70))
        
        # Return the screen as a numpy array if needed
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        if self.screen is not None:
            pygame.quit()


class VisionExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # After VecTransposeImage wrapper, images are already in [C, H, W] format
        # So input channels are the first dimension
        n_input_channels = observation_space.shape[0]
        
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(n_input_channels, 32, 8, stride=4),  # Larger kernel for full-screen input
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, 4, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 128, 3, stride=2),
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


# Create environment pipeline
def make_env():
    env = VisualSimulationEnv()
    env = Monitor(env)
    return env

env = DummyVecEnv([lambda: VisualSimulationEnv()])
# Use VecTransposeImage wrapper - it will convert from HWC to CHW format
env = VecTransposeImage(env)

# Configure model parameters
policy_kwargs = dict(
    features_extractor_class=VisionExtractor,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch=[128, 64]
)

# Initialize model
model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
    tensorboard_log=os.path.join('Training', 'Logs')
)


# Start training
model.learn(total_timesteps=500000)

# Evaluate model
eval_env = VisualSimulationEnv(render_mode='human')
mean_reward, _ = evaluate_policy(model, Monitor(eval_env), n_eval_episodes=10, render=True)

# Save model
model.save("ppo_visual_attention_full")

# # Load model (uncomment and change model name if you have a trained model)
# model = PPO.load("ppo_visual_attention_full")

# # For testing/evaluation
# env = VisualSimulationEnv(render_mode='human')
# env = Monitor(env)
# mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, render=True)
# print(f"Mean reward: {mean_reward:.2f}")

