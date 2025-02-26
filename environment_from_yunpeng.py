import gym
from gym import spaces
import numpy as np
import random
import pickle
import time
from torch.utils.tensorboard import SummaryWriter
import pygame

# ----- Pygame Settings -----
pygame.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Colors
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)

# Initial positions
red_dot_pos = [screen_width // 2, screen_height // 2]
blue_block_pos = [random.randint(0, screen_width - 50), random.randint(0, screen_height - 50)]
big_block_pos = [screen_width // 2 - 50, screen_height // 2 - 50]

# We’ll store the direction as an [dx, dy] vector pointing from red_dot_pos
line_direction = [0, -1]  # Default: Up


class VisualSimulationEnv(gym.Env):
    def __init__(self):
        super(VisualSimulationEnv, self).__init__()

        # --- ACTION SPACE ---
        # We will choose from 4 discrete directions:
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        self.action_space = spaces.Discrete(4)

        # --- OBSERVATION SPACE ---
        # For simplicity, let's keep track of:
        # [red_x, red_y, blue_x, blue_y]
        # If you want to observe the line direction too, you can extend this to shape=(6,)
        self.observation_space = spaces.Box(
            low=0,
            high=max(screen_width, screen_height),
            shape=(4,),
            dtype=np.float32
        )

        # --- Episode Variables ---
        self._ep_length = 50  # Max episode length
        self._steps = 0  # Current step

    def reset(self):
        global blue_block_pos, line_direction

        self._steps = 0

        # Reset the blue block to a random location
        blue_block_pos = [
            random.randint(0, screen_width - 50),
            random.randint(0, screen_height - 50)
        ]

        # Reset the line direction (optional)
        line_direction = [0, -1]  # Default back to Up

        # TODO these could be changed to self._get_obs() to avoid duplication
        return np.array([
            red_dot_pos[0],
            red_dot_pos[1],
            blue_block_pos[0],
            blue_block_pos[1],
        ], dtype=np.float32)

    def step(self, action):
        global line_direction

        self._steps += 1

        # Map discrete action to a direction “teleport”
        # 0 = Up, 1 = Right, 2 = Down, 3 = Left
        if action == 0:
            line_direction = [0, -1]
        elif action == 1:
            line_direction = [1, 0]
        elif action == 2:
            line_direction = [0, 1]
        elif action == 3:
            line_direction = [-1, 0]

        # Compute reward = the dot product between line_direction and direction to the blue block
        to_blue = [
            blue_block_pos[0] - red_dot_pos[0],
            blue_block_pos[1] - red_dot_pos[1]
        ]
        norm_to_blue = np.linalg.norm(to_blue)
        if norm_to_blue == 0:
            # If the blue block is exactly on the red dot, define some default
            reward = 0.0
        else:
            # Normalize to_blue
            to_blue[0] /= norm_to_blue
            to_blue[1] /= norm_to_blue
            reward = np.dot(line_direction, to_blue)

        # Define a success threshold
        done = (reward > 0.9)

        # NOTE early stopping if the episode length exceeds the limit
        if self._steps >= self._ep_length:
            done = True

        # Next observation
        obs = np.array([
            red_dot_pos[0],
            red_dot_pos[1],
            blue_block_pos[0],
            blue_block_pos[1],
        ], dtype=np.float32)

        return obs, reward, done, {}

    def render(self, mode='rgb_array'):
        screen.fill(BLACK)

        # Draw the red dot
        pygame.draw.circle(screen, RED, red_dot_pos, 10)

        # Draw the blue block
        pygame.draw.rect(screen, BLUE, (blue_block_pos[0], blue_block_pos[1], 50, 50))

        # Draw a big gray block (just an example)
        pygame.draw.rect(screen, GRAY, (big_block_pos[0], big_block_pos[1], 100, 100))

        # Draw the line indicating direction
        pygame.draw.line(
            screen,
            (255, 255, 255),
            red_dot_pos,
            [
                red_dot_pos[0] + line_direction[0] * 100,
                red_dot_pos[1] + line_direction[1] * 100
            ],
            2
        )
        if mode == 'rgb_array':
            return pygame.surfarray.array3d(screen)

    def close(self):
        pygame.quit()


# ----- Q-learning Example -----
env = VisualSimulationEnv()

alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 2000

# Q-table will map from “state” -> action values
# But the state is continuous in principle. For demo, we do a naive approach:
q_table = {}

writer = SummaryWriter('logs')


def discretize_state(observation):
    # Very rough discretization to make Q-table workable in this example
    # (In real usage, consider approximate methods or smaller states)
    return tuple((observation // 20).astype(int))  # bin each coordinate by 20 pixels


for episode in range(num_episodes):
    obs = env.reset()
    state = discretize_state(obs)
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        # If state not in q_table, initialize
        if state not in q_table:
            q_table[state] = np.zeros(env.action_space.n)

        # Epsilon-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_obs, reward, done, _ = env.step(action)
        next_state = discretize_state(next_obs)

        if next_state not in q_table:
            q_table[next_state] = np.zeros(env.action_space.n)

        # Update Q
        q_table[state][action] = (1 - alpha) * q_table[state][action] + \
                                 alpha * (reward + gamma * np.max(q_table[next_state]))

        state = next_state
        total_reward += reward
        step_count += 1

    # Write stats to TensorBoard
    writer.add_scalar('Total Reward', total_reward, episode)
    writer.add_scalar('Average Reward', total_reward / step_count, episode)
    writer.add_scalar('Epsilon', epsilon, episode)

    print(
        f"Episode {episode + 1}/{num_episodes} | Total Reward: {total_reward:.2f} | Steps: {step_count} | Epsilon: {epsilon:.2f}")

    # Decay epsilon
    epsilon = max(0.01, epsilon * 0.99)

env.close()
writer.close()

print("Training complete. To view logs, run:")
print("  tensorboard --logdir /home/baiy4/reader-agent-zuco/_draft_files/logs")
print("and open the provided URL in your browser.")
