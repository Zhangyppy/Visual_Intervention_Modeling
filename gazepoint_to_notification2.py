import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

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
BLOCK_SIZE = 50
MOVE_SPEED = 20
NOTIFICATION_POSITIONS = [
    (screen_width // 2, screen_height // 4),  # Top
    (3 * screen_width // 4, screen_height // 2),  # Right
    (screen_width // 2, 3 * screen_height // 4),  # Bottom
    (screen_width // 4, screen_height // 2),  # Left
]

#Reward
reward = 0



class VisualAttentionEnv(gym.Env):
    # metadata = {"render_modes": ["human"]}
    def __init__(self):

        super(VisualAttentionEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.observation_space = spaces.Box(
            low=0, high=max(screen_width, screen_height), shape=(4,), dtype=np.float32
        )  # [dot_x, dot_y, block_x, block_y]

        # Initialize state
        self.dot_pos = [screen_width // 2, screen_height // 2]
        self.block_pos = NOTIFICATION_POSITIONS[0]
        self.current_block_index = 0
        self.reward = 0
        self.steps = 0
        self.render_mode = None
        # Count the number of reached notification
        self.count = 0
        self.screen = None

    def reset(self, **kwargs):
        # Reset state
        self.dot_pos = [screen_width // 2, screen_height // 2]
        self.block_pos = NOTIFICATION_POSITIONS[self.current_block_index]
        self.reward = 0
        self.steps = 0

        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

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

        distance_dot_notification = (pow(self.dot_pos[0] - self.block_pos[0], 2) + pow(self.dot_pos[1] - self.block_pos[1], 2)) ** 0.5
        max_distance = (pow(screen_width, 2) + pow(screen_height, 2)) ** 0.5
        real_distance = (pow(screen_width // 2 - self.block_pos[0], 2) + pow(screen_height // 2 - self.block_pos[1], 2)) ** 0.5
        normalize_distance = (max_distance - distance_dot_notification) / max_distance
        #self.reward += normalize_distance

        # #calculate the x, y axis distance
        # distance_x = pow(self.dot_pos[0] - self.block_pos[0], 2) ** 0.5
        # distance_y = pow(self.dot_pos[1] - self.block_pos[1], 2) ** 0.5
        #
        # #normalize the distance of x, yaxis between dot and block
        # normalized_x = (screen_width - distance_x) / screen_width
        # normalized_y = (screen_height - distance_y) / screen_height

        # if (self.dot_pos[0] < screen_width - DOT_RADIUS): self.reward

        self.reward -= 0.1
        self.reward += normalize_distance

        # #Don't explore in the opposite direction
        # if real_distance < distance_dot_notification:
        #     self.reward -= real_distance - distance_dot_notification
        # else:
        #     self.reward +=

        #done = False

        if block_rect.colliderect(dot_rect):
            self.reward += 100
            self.count += 1
            self.current_block_index = (self.current_block_index + 1) % 4
            self.block_pos = NOTIFICATION_POSITIONS[self.current_block_index]
            self.dot_pos = [screen_width // 2, screen_height // 2]


        if self.count > 3:
            done = True

        # Check if the task should reset
        #done = self.steps >= 300
        truncated = False  # Gymnasium requires a truncated flag
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

    # def render(self, mode="human"):
    #     screen.fill(WHITE)
    #
    #     # Draw the red dot
    #     pygame.draw.circle(screen, RED, self.dot_pos, DOT_RADIUS)
    #
    #     # Draw the blue block
    #     pygame.draw.rect(
    #         screen,
    #         BLUE,
    #         (
    #             self.block_pos[0] - BLOCK_SIZE // 2,
    #             self.block_pos[1] - BLOCK_SIZE // 2,
    #             BLOCK_SIZE,
    #             BLOCK_SIZE,
    #         ),
    #     )
    #     #reward = self.reward
    #     # Display reward
    #     font = pygame.font.SysFont("Arial", 24)
    #     reward_text = font.render(f"Reward: {self.reward}", True, BLACK)
    #     screen.blit(reward_text, (10, 10))
    #
    #     pygame.display.flip()
    #     clock.tick(30)

    def render(self, mode="human"):
        pygame.display.flip()

    def close(self):
        pygame.quit()

# Custom callback to render the environment during training
class RenderCallback(BaseCallback):
    def __init__(self, env, render_freq: int = 10, verbose=0):
        super().__init__(verbose)
        self.env = env
        self.render_freq = render_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.render_freq == 0:
            self.env.render()
        return True

# Create environment and wrap it with Monitor for logging
log_dir = "logs/visual_attention/"
os.makedirs(log_dir, exist_ok=True)
env = DummyVecEnv([lambda: Monitor(VisualAttentionEnv(), log_dir)])
# Set up the model with tensorboard logging
model = PPO("MlpPolicy", env, device = "cpu", verbose=1, tensorboard_log=log_dir)

# Set up the EvalCallback to monitor the agent's performance during training
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=500,
    verbose=1,
)

# Set up the RenderCallback to render the environment during training
render_callback = RenderCallback(env, render_freq=10)

# Train the model
model.learn(total_timesteps=100000, callback=[eval_callback, render_callback])

# Save the trained model
model.save("visual_attention_model")

# Evaluate the policy after training
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward}")

# Close the environment
env.close()



