import datetime
import gc
import math
from gymnasium import spaces
import gymnasium as gym
import numpy as np
import pickle
import time
import pygame
import random
import os
import torch as th

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
from sb3_contrib import RecurrentPPO
from torch.utils.tensorboard import SummaryWriter

# Pygame settings
pygame.init()
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
GRID_SIZE = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Constants
AGENT_RADIUS = 5
BLOCK_SIZE = 40
MAX_MOVE_SPEED = 20  # Maximum speed for continuous actions
MIN_MOVE_SPEED = 5  # Minimum speed for movement when action is non-zero
MIN_BLOCK_GENERATION_DISTANCE = (
    50  # Minimum distance between agent and target to generate a block
)
ENVIRONMENT_BLOCK_SIZE = 20  # Size of block when on environment layer
NOTIFICATION_BLOCK_SIZE = 40  # Size of block when on notification layer

LAYER_ENVIRONMENT = 0
LAYER_NOTIFICATION = 1

# Check if CUDA or MPS is available
device = (
    "cuda"
    if th.cuda.is_available()
    else "mps" if th.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")
th.device(device)

# _detect_oscillation parameters for tuning
HISTORY_LENGTH = 10  # Track more positions for better pattern detection
MIN_MOVEMENT_THRESHOLD = (
    1.05 * MIN_MOVE_SPEED
)  # Min distance to be significant (NOTE: Once we figure out the reason why it refuse to move quicky, we can reduce this value)
OSCILLATION_THRESHOLD = (
    0.6  # Dot product threshold for direction change (-0.6 = ~120 degree turn)
)
MIN_REVERSALS = 3  # Number of direction reversals to detect oscillation
LOCAL_MINIMUM_RADIUS = 0.15 * MAX_MOVE_SPEED  # Radius to detect if stuck in local area
LOCAL_MIN_TIME_THRESHOLD = 8  # How many steps to be in local area to trigger
TARGET_PROGRESS_THRESHOLD = 0.02 * SCREEN_WIDTH  # Progress toward target threshold
COOLDOWN_STEPS = 5  # Steps to ignore oscillation detection after a trigger


class VisualSimulationEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(VisualSimulationEnv, self).__init__()

        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        # --- ACTION SPACE ---
        # Changed from discrete to continuous action space
        # Action is a 3D vector representing [x_direction, y_direction, layer]
        # First two values between -1 and 1 for movement, last value between 0 and 1 for layer selection
        # NOTE: The reason why the layer is not discrete is because the PPO do not support
        #       spaces.Tuple, so we use continuous action space for layer selection
        #       Then we use a threshold to convert the continuous action space to a discrete #       action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )

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
                "layer": spaces.Box(
                    low=np.array([0], dtype=np.int8),  # Environment layer
                    high=np.array([1], dtype=np.int8),  # Notification layer
                    shape=(
                        1,
                    ),  # Layer information as a float (0: environment, 1: notification)
                    dtype=np.int8,
                ),
            }
        )

        # --- Episode Variables ---
        self._ep_length = 100  # Max episode length
        self._steps = 0  # Current step

        # Initialize positions of agent and target
        self._initialize_positions()

        # Initialize layer state
        self.current_layer = LAYER_ENVIRONMENT  # Note: this assumes that the agent starts on the environment layer

        # Initialize pygame if in human render mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            self._init_render()

        # Initialize observation buffers
        self._obs_canvas = None
        self._visual_obs_buffer = None
        self._cached_target_boundaries = None

        # Initialize state tracking variables
        self.reward = 0
        self.reward_count = 0
        self.exploration_count = 0  # Track how many exploration rewards/penalties given
        self.action_history = np.full(
            10, 255, dtype=np.uint8
        )  # Reset action history with 255 (no action)
        self.visited_positions = set()
        self.position_history = [self.agent_pos.copy()]
        self.target_distance_history = np.zeros(10, dtype=np.float32)
        self.target_distance_history[0] = np.linalg.norm(
            self.agent_pos - self.target_pos
        )
        self.oscillation_cooldown = 0
        self.target_layer = None
        self.target_layer_block_size = None

    def _initialize_positions(self):
        """Initialize agent and target positions"""
        self.agent_pos = np.array(
            [
                random.randint(AGENT_RADIUS, SCREEN_WIDTH - AGENT_RADIUS),
                random.randint(AGENT_RADIUS, SCREEN_HEIGHT - AGENT_RADIUS),
            ],
            dtype=np.float32,
        )

        self.target_layer = random.randint(0, 1)
        self.target_layer_block_size = (
            ENVIRONMENT_BLOCK_SIZE
            if self.target_layer == LAYER_ENVIRONMENT
            else NOTIFICATION_BLOCK_SIZE
        )

        # Place the target at a random position, but ensure it's not right on top of the agent
        while True:
            self.target_pos = np.array(
                [
                    random.randint(
                        self.target_layer_block_size // 2,
                        SCREEN_WIDTH - self.target_layer_block_size // 2,
                    ),
                    random.randint(
                        self.target_layer_block_size // 2,
                        SCREEN_HEIGHT - self.target_layer_block_size // 2,
                    ),
                ],
                dtype=np.float32,
            )

            # Make sure the target is not too close to the agent at initialization
            # NOTE: beware if we scale down the screen size, this value needs to be scaled down as well
            if (
                np.linalg.norm(self.agent_pos - self.target_pos)
                > MIN_BLOCK_GENERATION_DISTANCE
            ):
                break

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Visual Attention Task")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility

        Returns:
            tuple: (observation, info)
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset positions
        self._initialize_positions()

        # Reset state tracking variables
        self.reward = 0
        self._steps = 0
        self.exploration_count = 0
        self.reward_count = 0
        self.action_history = np.full(
            10, 255, dtype=np.uint8
        )  # Reset action history with 255 (no action)
        self.position_history = [self.agent_pos.copy()]
        self.target_distance_history = np.zeros(10, dtype=np.float32)
        self.target_distance_history[0] = np.linalg.norm(
            self.agent_pos - self.target_pos
        )
        self.oscillation_cooldown = 0
        self.visited_positions = set()
        self.current_layer = LAYER_ENVIRONMENT

        # Update cached target boundaries for observation rendering
        if hasattr(self, "_cached_target_boundaries"):
            self._cached_target_boundaries = self._get_target_boundaries(
                self.target_layer_block_size
            )

        # Process and handle events to avoid pygame becoming unresponsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        return self._get_obs(), {}

    def _get_target_boundaries(self, block_size):
        """
        Calculate the target boundaries
        Returns:
            tuple: (x_min, y_min, x_max, y_max) of target boundaries
        """
        x_min = int(self.target_pos[0] - block_size // 2)
        y_min = int(self.target_pos[1] - block_size // 2)
        x_max = x_min + block_size
        y_max = y_min + block_size
        return x_min, y_min, x_max, y_max

    def _check_collision(self, block_size):
        """Check if the agent collides with the target"""
        # Calculate the center of the target
        target_center = self.target_pos

        # Check if the distance between the agent's center and the target's center
        # is less than the sum of the agent's radius and half the target size
        distance = np.linalg.norm(self.agent_pos - target_center)
        return distance < (AGENT_RADIUS + block_size / 2)

    def _detect_oscillation(self, action):
        """
        Enhanced oscillation detection that handles both local minimum (not moving enough)
        and looping behavior patterns.
        """
        # Skip detection during initial steps or during cooldown
        if self._steps < HISTORY_LENGTH:
            return False

        # Handle cooldown after detecting oscillation
        if hasattr(self, "oscillation_cooldown") and self.oscillation_cooldown > 0:
            self.oscillation_cooldown -= 1
            return False

        # Initialize position history if not already tracked
        if not hasattr(self, "position_history"):
            self.position_history = [self.agent_pos.copy()]  # Initialize as list
            self.target_distance_history = np.zeros(HISTORY_LENGTH, dtype=np.float32)
            self.oscillation_cooldown = 0

        # Update distance to target history
        current_target_distance = np.linalg.norm(self.agent_pos - self.target_pos)

        if not hasattr(self, "target_distance_history"):
            self.target_distance_history = np.zeros(HISTORY_LENGTH, dtype=np.float32)

        self.target_distance_history = np.roll(self.target_distance_history, 1)
        self.target_distance_history[0] = current_target_distance

        # Compute vectors between consecutive positions
        if len(self.position_history) < 2:
            return False

        # Always convert position_history to numpy array for calculations
        position_array = np.array(self.position_history[:HISTORY_LENGTH])

        # Compute vectors and magnitudes only once
        vectors = np.diff(position_array, axis=0)
        magnitudes = np.sqrt(np.sum(vectors**2, axis=1))

        # LOCAL MINIMUM DETECTION (not moving enough)
        # ---------------------------------------
        if len(position_array) >= LOCAL_MIN_TIME_THRESHOLD:
            # 1. Check if agent is staying within a small radius
            center = np.mean(position_array[:LOCAL_MIN_TIME_THRESHOLD], axis=0)
            distances_from_center = np.sqrt(
                np.sum(
                    (position_array[:LOCAL_MIN_TIME_THRESHOLD] - center) ** 2, axis=1
                )
            )

            # 2. Check if not making progress toward target
            target_progress = (
                self.target_distance_history[LOCAL_MIN_TIME_THRESHOLD - 1]
                - self.target_distance_history[0]
            )

            local_minimum_detected = (
                # All recent positions are within a small radius
                np.all(distances_from_center < LOCAL_MINIMUM_RADIUS)
                and
                # Not making meaningful progress toward target
                target_progress < TARGET_PROGRESS_THRESHOLD
                and
                # Agent is actively trying to move (non-zero action)
                np.any(np.abs(action[:2]) > 0.2)
            )

            if local_minimum_detected:
                # Set cooldown to prevent repeated triggers
                self.oscillation_cooldown = COOLDOWN_STEPS
                return True

        # MOVEMENT LOOPING DETECTION
        # ---------------------------------------
        # Only analyze significant movements
        significant_movements = magnitudes > MIN_MOVEMENT_THRESHOLD
        if np.sum(significant_movements[: MIN_REVERSALS + 1]) < MIN_REVERSALS + 1:
            return False

        # Check for oscillation patterns in recent movements
        # 1. Get normalized vectors for significant movements only
        significant_indices = np.where(significant_movements[: MIN_REVERSALS + 1])[0]
        if len(significant_indices) < MIN_REVERSALS + 1:
            return False

        # Pre-normalize all significant vectors at once
        significant_vectors = vectors[significant_indices]
        significant_magnitudes = magnitudes[significant_indices]

        # Avoid division by zero
        valid_magnitudes = significant_magnitudes > 0
        normalized_vectors = np.zeros_like(significant_vectors)
        normalized_vectors[valid_magnitudes] = (
            significant_vectors[valid_magnitudes]
            / significant_magnitudes[valid_magnitudes, np.newaxis]
        )

        # 2. Calculate dot products between consecutive vectors
        movement_pairs = min(len(normalized_vectors) - 1, MIN_REVERSALS + 1)
        direction_changes = 0

        for i in range(movement_pairs):
            if i + 1 < len(normalized_vectors):
                dot_product = np.sum(normalized_vectors[i] * normalized_vectors[i + 1])
                if dot_product < -OSCILLATION_THRESHOLD:
                    direction_changes += 1

        looping_detected = direction_changes >= MIN_REVERSALS

        if looping_detected:
            # Set cooldown to prevent repeated triggers
            self.oscillation_cooldown = COOLDOWN_STEPS
            return True

        # 3. Pattern detection - check if agent is revisiting same locations
        if len(position_array) >= HISTORY_LENGTH:
            # Create a spatial grid to detect revisited areas
            visited_areas = {}
            cell_size = (
                SCREEN_WIDTH / 10
            )  # Grid cell size (divide screen into 10x10 grid)

            # Count visits to each grid cell
            for pos in position_array:
                grid_x = int(pos[0] / cell_size)
                grid_y = int(pos[1] / cell_size)
                grid_key = (grid_x, grid_y)

                visited_areas[grid_key] = visited_areas.get(grid_key, 0) + 1

            # Check if any grid cell has been visited 3+ times
            revisited_cells = sum(1 for count in visited_areas.values() if count >= 3)

            # If we've revisited cells multiple times and not making progress toward target
            pattern_detected = (
                revisited_cells >= 2
                and target_progress < TARGET_PROGRESS_THRESHOLD
                and self._steps
                > 20  # Only apply this after agent has had time to explore
            )

            if pattern_detected:
                self.oscillation_cooldown = COOLDOWN_STEPS
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

        # Extract movement and layer actions from the combined action vector
        movement_action = action[:2]  # First two components for movement
        layer_action = action[2]  # Last component for layer selection

        # Convert layer_action from continuous [0,1] to discrete [0,1]
        self.current_layer = int(layer_action > 0.5)  # Threshold at 0.5

        # Update movement action history
        self.action_history = np.roll(self.action_history, 1)
        self.action_history[0] = int(
            np.argmax(np.abs(movement_action))
        )  # Store strongest direction for history

        # Normalize and scale the action vector
        # Actions are in range [-1, 1], scale them by MAX_MOVE_SPEED
        action_vector = np.array(movement_action, dtype=np.float32)
        vector_magnitude = np.linalg.norm(action_vector)

        # Normalize only if the magnitude is not zero to avoid division by zero
        if vector_magnitude > 1e-6:
            # Normalize to unit vector
            normalized_action = action_vector / vector_magnitude
            speed = max(vector_magnitude * MAX_MOVE_SPEED, MIN_MOVE_SPEED)
            move_vector = normalized_action * speed
        else:
            # Zero movement
            move_vector = np.zeros(2, dtype=np.float32)

        # Calculate the intended (unclipped) position
        intended_pos = self.agent_pos + move_vector

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

        # Update position history
        self.position_history.insert(0, self.agent_pos.copy())
        # Keep only the most recent positions to avoid memory issues
        if len(self.position_history) > 10:  # Keep last 10 positions
            self.position_history = self.position_history[:10]

        # --- Reward Structure ---
        reward = -0.1

        # Boundary penalty
        if hit_boundary:
            reward -= 10

        # Terminal reward for reaching target (only if on same layer)
        if self._check_collision(self.target_layer_block_size):
            # Only give reward if on the correct layer otherwise it will be punished
            if self.current_layer == self.target_layer:
                reward += 200
                self.reward_count += 1
            else:
                reward -= 10
            if self.reward_count >= 3:
                done = True
        # else:
        # NOTE: The current goal is to make the agent to stay on target layer
        # However, sometimes the target layer is the environment layer, is this really match the real world?
        #     if self.current_layer != LAYER_ENVIRONMENT:
        #         reward -= 0.5

        # Reward for exploring (changing actions)
        if not self._detect_oscillation(movement_action):
            if len(self.position_history) > 3:
                recent_movement = self.agent_pos - self.position_history[0]
                recent_movement_norm = np.linalg.norm(recent_movement)

                if recent_movement_norm > MIN_MOVE_SPEED:
                    # Normalize the movement vector (only once)
                    recent_dir = recent_movement / recent_movement_norm

                    # Check if the direction has changed
                    direction_changed = False
                    for i in range(1, min(3, len(self.position_history) - 1)):
                        prev_movement = (
                            self.position_history[i - 1] - self.position_history[i]
                        )
                        prev_movement_norm = np.linalg.norm(prev_movement)

                        if prev_movement_norm > MIN_MOVE_SPEED:
                            prev_dir = prev_movement / prev_movement_norm
                            # If dot product is less than threshold, directions are different
                            if np.dot(recent_dir, prev_dir) < 0.8:  # Not too similar
                                direction_changed = True
                                break

                    if direction_changed:
                        reward += 0.2
        else:
            reward -= 1

        # Reward for exploring new positions, use discretized position for our convience
        # Discretize position to track visited areas
        grid_x = int(self.agent_pos[0] // GRID_SIZE)
        grid_y = int(self.agent_pos[1] // GRID_SIZE)
        grid_pos = (grid_x, grid_y)

        if grid_pos not in self.visited_positions:
            self.visited_positions.add(grid_pos)
            # diminishing reward for finding new positions
            reward += 1 * math.exp(len(self.visited_positions) * -0.05)

        # Check if episode is done due to step limit
        if self._steps >= self._ep_length:
            done = True

        # Store the reward for rendering purposes
        self.reward += reward

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        # Create a canvas for the full screen observation or reuse existing one
        if not hasattr(self, "_obs_canvas") or self._obs_canvas is None:
            self._obs_canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self._visual_obs_buffer = np.empty(
                (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
            )
            self._cached_target_boundaries = self._get_target_boundaries(
                self.target_layer_block_size
            )

        # Reuse canvas to avoid memory allocation each time
        canvas = self._obs_canvas
        canvas.fill(WHITE)

        # Draw the blue target block using cached boundaries
        target_x_min, target_y_min, _, _ = self._cached_target_boundaries
        pygame.draw.rect(
            canvas,
            BLUE,
            (
                target_x_min,
                target_y_min,
                self.target_layer_block_size,
                self.target_layer_block_size,
            ),
        )

        # Draw the red agent
        pygame.draw.circle(canvas, RED, self.agent_pos.astype(int), AGENT_RADIUS)

        # draw the space boundaries outer boundary
        pygame.draw.rect(canvas, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        # Get pixel data into our buffer - more efficient than creating new array
        # PyGame returns in (W, H, C) format, convert to (H, W, C)
        pygame.pixelcopy.surface_to_array(
            np.transpose(self._visual_obs_buffer, (1, 0, 2)), canvas
        )

        # Return both visual observation and position information
        return {
            "visual": self._visual_obs_buffer,
            "position": self.agent_pos.astype(np.float32),
            "layer": np.array(
                [self.current_layer], dtype=np.int8
            ),  # Add current layer observation
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
        target_x_min, target_y_min, _, _ = self._get_target_boundaries(
            self.target_layer_block_size
        )
        pygame.draw.rect(
            self.screen,
            BLUE,
            (
                target_x_min,
                target_y_min,
                self.target_layer_block_size,
                self.target_layer_block_size,
            ),
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

        # Display current layer
        layer_text = font.render(
            f"Layer: {'Environment' if self.current_layer == LAYER_ENVIRONMENT else 'Notification'}",
            True,
            BLACK,
        )
        self.screen.blit(layer_text, (10, 70))

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
        self.screen.blit(history_text, (10, 90))

        # Display exploration count
        exploration_text = font.render(
            f"Explored: {self.exploration_count}",
            True,
            BLACK,
        )
        self.screen.blit(exploration_text, (10, 110))

        # Return the screen as a numpy array if needed
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

    def close(self):
        """
        Clean up resources used by the environment.
        """
        # Free pygame resources
        if hasattr(self, "screen") and self.screen is not None:
            pygame.display.quit()  # Properly quit the display
            pygame.quit()

        # Explicitly free the observation canvas to avoid memory leaks
        if hasattr(self, "_obs_canvas"):
            self._obs_canvas = None
            del self._obs_canvas

        # Free any other buffers
        if hasattr(self, "_visual_obs_buffer"):
            del self._visual_obs_buffer

        if hasattr(self, "_cached_target_boundaries"):
            del self._cached_target_boundaries

        # Clear history collections
        if hasattr(self, "position_history"):
            self.position_history = []

        if hasattr(self, "target_distance_history"):
            self.target_distance_history = None

        # Clear visited positions
        if hasattr(self, "visited_positions"):
            self.visited_positions.clear()

        # Clear action history
        if hasattr(self, "action_history"):
            self.action_history = None


class CoordConv(th.nn.Module):
    """
    Coordinate Convolutional Layer
    Source: https://paperswithcode.com/method/coordconv

    Adds coordinate channels to the input tensor before applying convolution.
    Can optionally include a radial distance channel.
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        with_r=False,
    ):
        super(CoordConv, self).__init__()
        # Add +2 for coordinate channels, +1 more if using radial distance
        additional_channels = 3 if with_r else 2
        self.with_r = with_r
        self.conv = th.nn.Conv2d(
            input_channels + additional_channels,
            output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x):
        batch_size, _, height, width = x.shape

        # Create coordinate channels
        y_coords = (
            th.linspace(-1, 1, height)
            .view(1, 1, height, 1)
            .expand(batch_size, 1, height, width)
        )
        x_coords = (
            th.linspace(-1, 1, width)
            .view(1, 1, 1, width)
            .expand(batch_size, 1, height, width)
        )

        # Move to the same device as input
        coords = th.cat([x_coords, y_coords], dim=1).to(x.device)

        # Add radial distance channel if requested
        if self.with_r:
            # Calculate radial distance (Euclidean distance from center)
            r_coords = th.sqrt(y_coords**2 + x_coords**2).to(x.device)
            coords = th.cat([coords, r_coords], dim=1)

        # Concatenate coordinates with input features
        x_with_coords = th.cat([x, coords], dim=1)

        return self.conv(x_with_coords)


class VisionExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 256,
        input_channels=None,
        resize_to=None,
    ):
        # Initialize with the visual part of the observation space
        super().__init__(observation_space, features_dim)

        # Store the resize dimensions
        self.resize_to = resize_to
        if resize_to is not None:
            print(
                f"Visual input will be resized from {SCREEN_WIDTH}x{SCREEN_HEIGHT} to {resize_to[0]}x{resize_to[1]}"
            )

        # Extract shapes from the observation space
        # The format depends on whether the CustomVecTranspose was applied or not
        visual_space = observation_space["visual"]

        # If input_channels is specified, use it (for frame stacking)
        if input_channels is not None:
            n_input_channels = input_channels
            print(f"Using provided input channels: {n_input_channels}")
        else:
            # Determine the correct channel dimension
            if (
                len(visual_space.shape) == 3
            ):  # (C, H, W) format after CustomVecTranspose
                n_input_channels = visual_space.shape[0]
            else:  # (H, W, C) format before CustomVecTranspose
                n_input_channels = visual_space.shape[2]

        pos_dim = observation_space["position"].shape[0]
        # action_history_dim = observation_space["action_history"].shape[0]

        print(f"Initializing CNN with {n_input_channels} input channels")

        # Add a resize layer if resize_to is specified
        if resize_to is not None:
            if th.cuda.is_available():
                self.resize = th.nn.Sequential(
                    th.nn.Upsample(size=resize_to, mode="bilinear", align_corners=False)
                )
            else:
                # Without CUDA, use a simpler approach
                self.resize = lambda x: th.nn.functional.interpolate(
                    x, size=resize_to, mode="bilinear", align_corners=False
                )

        self.cnn = th.nn.Sequential(
            # th.nn.Conv2d(n_input_channels, 32, 8, stride=4),
            CoordConv(
                # th.nn.Conv2d(
                n_input_channels,
                output_channels=8,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
            ),
            th.nn.LeakyReLU(),
            th.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(2, 2),
            ),
            th.nn.LeakyReLU(),
            th.nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(2, 2),
            ),
            th.nn.LeakyReLU(),
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
                        sample = sample.permute(2, 0, 1).unsqueeze(
                            0
                        )  # Convert and add batch

            print(f"Original sample shape: {sample.shape}")  # Should be (1, C, H, W)
            # Resize the sample if resize_to is specified
            if self.resize_to is not None:
                resized_sample = self.resize(sample)
                print(
                    f"Resized sample shape: {resized_sample.shape}"
                )  # Should be (1, C, resize_h, resize_w)
            else:
                resized_sample = sample
            n_flatten = self.cnn(resized_sample).shape[1]
            print(f"Flattened features size: {n_flatten}")

        # Create a network for processing position data
        self.pos_net = th.nn.Sequential(th.nn.Linear(pos_dim, 64), th.nn.ReLU())

        # Create a network for processing action history
        # self.action_history_net = th.nn.Sequential(
        #     th.nn.Linear(action_history_dim, 32), th.nn.ReLU()
        # )

        # Final layer to combine all features
        # self.combined = th.nn.Sequential(
        #     th.nn.Linear(n_flatten + 64 + 32, features_dim), th.nn.ReLU()
        # )

        self.combined = th.nn.Sequential(
            # th.nn.Linear(n_flatten + 64, features_dim),
            # th.nn.LayerNorm(features_dim),
            # th.nn.ReLU(),
            th.nn.Linear(n_flatten + 64, features_dim),
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
                if (
                    not hasattr(self, "_visual_tensor_buffer")
                    or self._visual_tensor_buffer.shape[1:]
                    != observations["visual"].shape
                ):
                    shape = observations["visual"].shape
                    self._visual_tensor_buffer = th.empty(
                        (1,) + shape if len(shape) == 3 else shape,
                        dtype=th.float32,
                        device="cuda",
                    )

                # Copy directly into the buffer
                visual_obs = self._visual_tensor_buffer
                visual_obs.copy_(
                    th.as_tensor(
                        observations["visual"], dtype=th.float32, device="cuda"
                    ),
                    non_blocking=True,
                )
            else:
                visual_obs = th.as_tensor(observations["visual"], dtype=th.float32)

        # Add batch dimension (CHW -> NCHW)
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        # Apply resize function if needed
        if self.resize_to is not None:
            visual_obs = self.resize(visual_obs)

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
                pos_obs = th.as_tensor(
                    observations["position"], dtype=th.float32, device="cuda"
                )
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

        print(
            f"Original visual shape: {orig_visual_shape}, detected {n_channels} channels"
        )

        # Update observation space to handle dictionary
        self.observation_space = spaces.Dict(
            {
                "visual": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        n_channels,
                        SCREEN_HEIGHT,
                        SCREEN_WIDTH,
                    ),  # CHW format with stacked frames
                    dtype=np.uint8,
                ),
                "position": venv.observation_space[
                    "position"
                ],  # Keep position space as is
                "layer": venv.observation_space[
                    "layer"
                ],  # Keep layer space as is (already stacked)
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
            layer = obs["layer"]  # Shape: (n_stack,)
        else:
            # For vectorized observations (e.g., from DummyVecEnv)
            # This is the most common case during training
            visual_obs = obs[0]["visual"]  # Shape: (H, W, C) or (1, H, W, C)
            position = obs[0]["position"]  # Shape: (2,) or (1, 2)
            # action_history = obs[0]["action_history"]  # Shape: (10,) or (1, 10)
            layer = obs[0]["layer"]  # Shape: (n_stack,) or (1, n_stack)

        # Single fast check for both visual and position to remove batch dim
        # Remove any extra batch dimension from DummyVecEnv if present
        if isinstance(visual_obs, np.ndarray) and len(visual_obs.shape) == 4:
            visual_obs = visual_obs[0]  # Convert (1, H, W, C) to (H, W, C)

        if isinstance(position, np.ndarray) and len(position.shape) == 2:
            position = position[0]  # Convert (1, 2) to (2,)

        if isinstance(layer, np.ndarray):
            if len(layer.shape) > 1:
                layer = layer[0]  # Convert (1, n_stack) to (n_stack,)
            if len(layer.shape) == 0:
                layer = layer.reshape(1)  # Ensure shape is (1,)

        # Convert from HWC to CHW format - only once and only if needed
        # Check if already in CHW format to avoid unnecessary transpose
        if (
            isinstance(visual_obs, np.ndarray)
            and len(visual_obs.shape) == 3
            and visual_obs.shape[2] <= 12
        ):
            # Standard case - needs transpose

            h, w, c = visual_obs.shape

            # Create or resize transpose buffer if needed
            if not hasattr(
                self, "_transpose_buffer"
            ) or self._transpose_buffer.shape != (c, h, w):
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
            "layer": layer,
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
    # Training parameters
    total_timesteps = 1000000
    check_freq = 50000  # Save checkpoint every 50k steps
    MODEL_TAG = "PPO_Layer_shaping_Continuous"

    # Create environment with frame stacking
    n_stack = 4  # Stack 4 frames
    env, input_channels = create_env_with_frame_stacking(n_stack=n_stack)

    # Configure model parameters
    policy_kwargs = dict(
        features_extractor_class=VisionExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            # Pass the correct number of input channels for the stacked frames
            input_channels=input_channels,
            # Downscale the visual input to this size (height, width)
            # resize_to=(84, 84)
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
        batch_size=512,
        ent_coef=0.001,
        tensorboard_log=os.path.join("Training", "Logs"),
        device=device,
        target_kl=0.1,
    )

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
        if hasattr(th, "set_num_threads"):
            num_threads = max(8, os.cpu_count() // 2)
            th.set_num_threads(num_threads)
            print(f"Setting PyTorch threads to {num_threads}")

        # Configure PyTorch to release memory more aggressively
        if hasattr(th.cuda, "empty_cache"):
            th.cuda.empty_cache()

        model.learn(
            total_timesteps=remaining_timesteps,
            progress_bar=True,
            callback=callback_list,
            tb_log_name=MODEL_TAG,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final checkpoint...")
        # Run garbage collection before saving to reduce memory footprint
        gc.collect()
        if th.cuda.is_available():
            th.cuda.empty_cache()

        save_checkpoint(
            model, start_timestep + model.num_timesteps, model_tag=MODEL_TAG
        )
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
    print(
        "This will handle the observation stacking and preprocessing correctly for visual rendering."
    )
