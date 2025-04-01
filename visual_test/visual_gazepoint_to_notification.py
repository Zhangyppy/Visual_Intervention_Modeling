import datetime
import gc
import math
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
import numpy as np
import pickle
import time
import pygame
import random
import os
import torch as th
from cbam import CBAM
from target import Target, LAYER_ENVIRONMENT, LAYER_NOTIFICATION

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
GREEN = (0, 255, 0)
NOTIFICATION_COLORS = BLUE
ENVIRONMENT_COLORS = GREEN
BLUE_TEXTURE = (70, 70, 220)
GREEN_TEXTURE = (0, 100, 0)

# Constants
AGENT_RADIUS = 10
ENVIRONMENT_BLOCK_SIZE = 30  # Size of block when on environment layer
NOTIFICATION_BLOCK_SIZE = 50  # Size of block when on notification layer
MAX_MOVE_SPEED = 30  # Maximum speed for continuous actions
MIN_MOVE_SPEED = 1  # Minimum speed for movement when action is non-zero
MIN_BLOCK_GENERATION_DISTANCE = (
    max(NOTIFICATION_BLOCK_SIZE + 10, 60)  # Minimum distance between agent and target to generate a block
)
BOUNDARY_PENALTY = 1  # Penalty for hitting the boundary
COMPLETION_REWARD = 100  # Reward for completing all expirable targets

LAYER_ENVIRONMENT = 0
LAYER_NOTIFICATION = 1

# Check if CUDA or MPS is available
device = (
    "cuda"
    if th.cuda.is_available()
    # else "mps" if th.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")
th.device(device)
device = th.device(device)
th.set_default_device(device) if hasattr(th, "set_default_device") else None

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
        #       Then we use a threshold to convert the continuous action space to a discrete
        #       action space
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
                "current_layer": spaces.Box(
                    low=np.array([0], dtype=np.int8),  # Environment layer
                    high=np.array([1], dtype=np.int8),  # Notification layer
                    shape=(
                        1,
                    ),  # Layer information as a float (0: environment, 1: notification)
                    dtype=np.int8,
                ),
                "persistent_target_layer": spaces.Box(
                    low=np.array([0], dtype=np.int8),
                    high=np.array([1], dtype=np.int8),
                    shape=(1,),
                    dtype=np.int8,
                ),
                "expirable_target_layer": spaces.Box(
                    low=np.array([-1], dtype=np.int8),  # -1 means no expirable target
                    high=np.array([1], dtype=np.int8),
                    shape=(1,),
                    dtype=np.int8,
                ),
            }
        )

        # --- Episode Variables ---
        self._steps = 0  # Current step
        self.max_expirable_targets = 2  # Maximum number of expirable targets per episode, usually we give more reward to these targets
        self.expirable_target_count = 0

        # Target generation parameters
        self.min_target_generation_gap = (
            10  # Minimum steps between generating new expirable targets
        )
        self.target_generation_probability = (
            0.4  # Probability of generating a new target when eligible
        )
        self.last_target_generation_step = 0  # Step when the last target was generated
        self.has_expirable_target = False  # Flag to track if there's an active expirable target, NOTE: also a temporary solution that just consider 1 expirable target

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
        self.target_distance_history[0] = 0
        self.oscillation_cooldown = 0
        self.previous_layer = LAYER_ENVIRONMENT

    def _initialize_positions(self):
        """Initialize agent and target positions"""
        self.agent_pos = np.array(
            [
                random.randint(AGENT_RADIUS, SCREEN_WIDTH - AGENT_RADIUS),
                random.randint(AGENT_RADIUS, SCREEN_HEIGHT - AGENT_RADIUS),
            ],
            dtype=np.float32,
        )

        # Create the persistent target (never expires, lower reward)
        self.persistent_target = self._create_random_target(
            expires=False,
            reward=5,
            required_steps=1,
            color=BLUE,
            size=ENVIRONMENT_BLOCK_SIZE,
            layer=LAYER_ENVIRONMENT,
        )

        # Initialize target attribute (this will be set later in _update_expireable_targets)
        # but we need it as a placeholder for observation and distance calculation
        self.expirable_target = None

    def _create_random_target(
        self,
        expires=True,
        reward=40,
        required_steps=3,
        max_lifetime=30,
        size=None,
        color=None,
        layer=None,
    ):
        """
        Create a random target with specified properties
        Place the target at a random position, but ensure it's not right on top of the agent
        """
        while True:
            target_pos = np.array(
                [
                    random.randint(
                        size // 2,
                        SCREEN_WIDTH - size // 2,
                    ),
                    random.randint(
                        size // 2,
                        SCREEN_HEIGHT - size // 2,
                    ),
                ],
                dtype=np.float32,
            )

            # Make sure the target is not too close to the agent
            # NOTE: beware if we scale down the screen size, this value needs to be scaled down as well
            if (
                np.linalg.norm(self.agent_pos - target_pos)
                > MIN_BLOCK_GENERATION_DISTANCE
            ):
                # If we already have a persistent target, also ensure the new target isn't too close to it
                if (
                    not hasattr(self, "persistent_target")
                    or np.linalg.norm(self.persistent_target.pos - target_pos)
                    > MIN_BLOCK_GENERATION_DISTANCE
                ):
                    break

        # Create the target with provided settings
        return Target(
            pos=target_pos,
            required_steps=required_steps,
            reward=reward,
            expires=expires,
            max_lifetime=max_lifetime,
            size=size,
            color=color,
            layer=layer,
        )

    def _init_render(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Visual Attention Task")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset (compatibility with newer Gymnasium versions)

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
        self.expirable_target_count = 0
        self.has_expirable_target = False
        self.last_target_generation_step = 0
        self.action_history = np.full(
            10, 255, dtype=np.uint8
        )  # Reset action history with 255 (no action)
        self.position_history = [self.agent_pos.copy()]
        self.target_distance_history = np.zeros(10, dtype=np.float32)
        # Initialize target_distance_history with distance to persistent target
        self.target_distance_history[0] = np.linalg.norm(
            self.agent_pos - self.persistent_target.pos
        )
        self.oscillation_cooldown = 0
        self.visited_positions = set()
        self.current_layer = LAYER_ENVIRONMENT

        # Process and handle events to avoid pygame becoming unresponsive
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        return self._get_obs(), {}

    # def _check_collision(self):
    #     """Check if the agent collides with the target"""
    #     return self.target.check_collision(self.agent_pos, AGENT_RADIUS)

    def _detect_oscillation(self, action):
        """
        Enhanced oscillation detection that handles both local minimum (not moving enough)
        and looping behavior patterns.
        NOTE: outdated
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
        current_target_distance = np.linalg.norm(
            self.agent_pos - self.expirable_target.pos
        )

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
        terminated = False

        # Extract movement and layer actions from the combined action vector
        movement_action = action[:2]  # First two components for movement
        layer_action = action[2]  # Last component for layer selection

        prev_layer = self.current_layer
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

        # Check collisions and update targets
        reward += self._handle_target_interactions()

        # Update target lifetimes and generate new targets if needed
        self._update_expirable_targets()

        # Termination for completing all expirable targets
        if self.expirable_target_count >= self.max_expirable_targets and not self.has_expirable_target:
            terminated = True
            reward += COMPLETION_REWARD
            self.reward += COMPLETION_REWARD

        # Reward for exploring new positions, use discretized position for our convience
        # Discretize position to track visited areas
        grid_x = int(self.agent_pos[0] // GRID_SIZE)
        grid_y = int(self.agent_pos[1] // GRID_SIZE)
        grid_pos = (grid_x, grid_y)

        if grid_pos not in self.visited_positions:
            self.visited_positions.add(grid_pos)
            # diminishing reward for finding new positions
            reward += 1 * math.exp(len(self.visited_positions) * -0.05)

        if hit_boundary:
            reward = 0
            reward -= BOUNDARY_PENALTY

        # Store the reward for rendering purposes
        self.reward += reward

        # Update previous layer
        self.previous_layer = self.current_layer

        return self._get_obs(), reward, terminated, False, {}
    
    def _is_agent_switched_layer(self):
        return self.previous_layer != self.current_layer

    def _handle_target_interactions(self):
        """Handle interactions with both persistent and expirable targets"""
        reward = 0

        # Check persistent target first
        if self.persistent_target.check_collision(self.agent_pos, AGENT_RADIUS):
            # small reward for agent exploring new layer
            if self._is_agent_switched_layer():
                reward += self.persistent_target.reward / 10

            agent_on_correct_layer = self.current_layer == self.persistent_target.layer
            if agent_on_correct_layer:
                # Always give reward for persistent target (lower reward)
                reward += self.persistent_target.reward
            else:
                reward -= self.persistent_target.reward / 3

        # Check expirable target if it exists
        if self.has_expirable_target and self.expirable_target.check_collision(
            self.agent_pos, AGENT_RADIUS
        ):
            # small reward for agent exploring new layer
            if self._is_agent_switched_layer():
                reward += 0.5

            # Check if agent is on the correct layer
            agent_on_correct_layer = self.current_layer == self.expirable_target.layer
            target_completed = self.expirable_target.update_progress(
                True, agent_on_correct_layer
            )

            if agent_on_correct_layer:
                reward += self.expirable_target.reward
                self.reward_count += 1
            else:
                reward -= self.expirable_target.reward / 3

            # Immediately update has_expirable_target if the target is completed
            if target_completed:
                reward += self.expirable_target.reward # extra reward for completing the target
                self.has_expirable_target = False

        elif self.has_expirable_target:
            # Update target progress (not on target)
            self.expirable_target.update_progress(False, False)

        return reward

    def _update_expirable_targets(self):
        """Update expirable target lifetimes and generate new ones if needed"""
        # TODO: this is just a temporary solution that only support 1 expirable target
        # we should update this to support multiple expirable targets if the task requires it

        # If we don't have an active expirable target, consider generating one
        if not self.has_expirable_target:
            # Check if we've passed the minimum generation gap and haven't reached max targets
            steps_since_last = self._steps - self.last_target_generation_step

            # randomize the layer of the expirable target
            layer = random.randint(0, 1)

            if (
                steps_since_last >= self.min_target_generation_gap
                and self.expirable_target_count < self.max_expirable_targets
                and random.random() < self.target_generation_probability
            ):
                self.expirable_target = self._create_random_target(
                    expires=True,
                    reward=20,
                    required_steps=3,
                    max_lifetime=60,
                    color=GREEN,
                    size=(
                        NOTIFICATION_BLOCK_SIZE
                        if layer == LAYER_NOTIFICATION
                        else ENVIRONMENT_BLOCK_SIZE
                    ),
                    layer=layer,
                )
                self.expirable_target_count += 1
                self.last_target_generation_step = self._steps
                self.has_expirable_target = True
        else:
            # Update existing expirable target's lifetime
            target_expired = self.expirable_target.update_lifetime()

            # If the target expired, mark it as inactive
            if target_expired:
                self.has_expirable_target = False

    def _draw_target(self, surface, target: Target, is_render_mode=False):
        """
        Helper method to draw a target on a surface

        Args:
            surface: pygame surface to draw on
            target: Target object to draw
            is_render_mode: Whether this is being called from render mode (affects line thickness)
        """
        # Draw the target block
        target_x_min, target_y_min, _, _ = target.boundaries
        pygame.draw.rect(
            surface,
            target.color,
            (
                target_x_min,
                target_y_min,
                target.size,
                target.size,
            ),
        )

        # Add texture to the target
        line_spacing = target.size // 4
        line_thickness = 2

        if target.layer == LAYER_ENVIRONMENT:
            # Horizontal lines for environment targets
            for i in range(1, 4):
                pygame.draw.line(
                    surface,
                    target.texture,
                    (target_x_min, target_y_min + i * line_spacing),
                    (target_x_min + target.size, target_y_min + i * line_spacing),
                    line_thickness,
                )
        else:  # LAYER_NOTIFICATION
            # Vertical lines for notification targets
            for i in range(1, 4):
                pygame.draw.line(
                    surface,
                    target.texture,
                    (target_x_min + i * line_spacing, target_y_min),
                    (target_x_min + i * line_spacing, target_y_min + target.size),
                    line_thickness,
                )

    def _get_obs(self):
        # Create a canvas for the full screen observation or reuse existing one
        if not hasattr(self, "_obs_canvas") or self._obs_canvas is None:
            self._obs_canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            self._visual_obs_buffer = np.empty(
                (SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8
            )

        # Reuse canvas to avoid memory allocation each time
        canvas = self._obs_canvas
        canvas.fill(WHITE)

        # Draw the persistent target
        self._draw_target(canvas, self.persistent_target)

        # Draw the expirable target if it exists
        if self.has_expirable_target:
            self._draw_target(canvas, self.expirable_target)

        # Draw the red agent
        pygame.draw.circle(canvas, RED, self.agent_pos.astype(int), AGENT_RADIUS)

        # draw the space boundaries outer boundary
        pygame.draw.rect(canvas, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        # Get pixel data into our buffer - more efficient than creating new array
        # PyGame returns in (W, H, C) format, convert to (H, W, C)
        pygame.pixelcopy.surface_to_array(
            np.transpose(self._visual_obs_buffer, (1, 0, 2)), canvas
        )

        persistent_layer = np.array([self.persistent_target.layer], dtype=np.int8)
        if self.has_expirable_target:
            expirable_layer = np.array([self.expirable_target.layer], dtype=np.int8)
        else:
            expirable_layer = np.array([-1], dtype=np.int8)

        # Return both visual observation and position information
        return {
            "visual": self._visual_obs_buffer,
            "position": self.agent_pos.astype(np.float32),
            "current_layer": np.array([self.current_layer], dtype=np.int8),
            "persistent_target_layer": persistent_layer,
            "expirable_target_layer": expirable_layer,
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

        # Draw the persistent target
        self._draw_target(self.screen, self.persistent_target, is_render_mode=True)

        # Draw the expirable target if it exists
        if self.has_expirable_target:
            self._draw_target(self.screen, self.expirable_target, is_render_mode=True)

        # Draw the red agent
        pygame.draw.circle(self.screen, RED, self.agent_pos.astype(int), AGENT_RADIUS)

        # draw the space boundaries outer boundary
        pygame.draw.rect(self.screen, BLACK, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 5)

        # Return the screen as a numpy array if needed
        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)

        # Draw text info only in human render mode
        if self.render_mode == "human":
            font = pygame.font.SysFont("Arial", 14)

            # Display step count
            steps_text = font.render(
                f"Steps: {self._steps}", True, BLACK
            )
            self.screen.blit(steps_text, (10, 10))

            # Display distance to expirable target if it exists
            if self.has_expirable_target:
                distance = np.linalg.norm(self.agent_pos - self.expirable_target.pos)
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

            # Display expirable target info
            if self.has_expirable_target:
                target_text = font.render(
                    f"Expirable Target: {'Environment' if self.expirable_target.layer == LAYER_ENVIRONMENT else 'Notification'} "
                    f"({self.expirable_target.current_steps}/{self.expirable_target.required_steps}, Life: {self.expirable_target.lifetime}/{self.expirable_target.max_lifetime})",
                    True,
                    BLACK,
                )
                self.screen.blit(target_text, (10, 130))
            else:
                target_text = font.render("No active expirable target", True, BLACK)
                self.screen.blit(target_text, (10, 130))

            persistent_text = font.render(
                f"Persistent Target: {'Environment' if self.persistent_target.layer == LAYER_ENVIRONMENT else 'Notification'}",
                True,
                BLACK,
            )
            self.screen.blit(persistent_text, (10, 150))

            # Display count of generated targets
            targets_text = font.render(
                f"Targets: {self.expirable_target_count}/{self.max_expirable_targets}",
                True,
                BLACK,
            )
            self.screen.blit(targets_text, (10, 170))

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
        n_stack=1,
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
        layer_dim = observation_space["current_layer"].shape[0]
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

        """
        CNN overhual plan:
        Why update the CNN architecture?
        1. Increase initial kernel from 3x3 to 7x7 (from 9 pixels to 49 pixels)
            Our images are quite large and sometimes we even stack multiple frames. 
            Thus a larger kernel can enlarges receptive field to capture broader spatial patterns
            maybe it also helps detection of motion across stacked frames
        2. Implemente graduated kernel sizing (7→5→3→3)
            hierarchical feature extraction (global→local)
            more like biological vision processing
        3. Add more padding to final conv layer
            prevents information loss at image boundaries (reduces tunnel vision!!!! We have very serious tunnel vision problem)

        Downside? Its much slower than before. Others? We need to test it out.
        """
        self.cnn = th.nn.Sequential(
            # th.nn.Conv2d(n_input_channels, 32, 8, stride=4),
            CoordConv(
                # th.nn.Conv2d(
                n_input_channels,
                output_channels=8,
                kernel_size=(5, 5),
                stride=(2, 2),
                padding=(3, 3),
            ),
            th.nn.LeakyReLU(),
            # CBAM(gate_channels=8, reduction_ratio=2),
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

        # Create a network for processing position data (with frame stacking)
        self.pos_net = th.nn.Sequential(
            th.nn.Linear(pos_dim, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, 32),
            th.nn.ReLU(),
        )
        print(
            f"Position network output size: {self.pos_net(th.zeros((1, pos_dim))).shape}"
        )

        # Create a network for processing layer data (binary input with frame stacking)
        self.curr_layer_net = th.nn.Sequential(
            th.nn.Linear(layer_dim, 32),
            th.nn.ReLU(),
            th.nn.Linear(32, 16),
            th.nn.ReLU(),
        )
        print(
            f"Layer network output size: {self.curr_layer_net(th.zeros((1, layer_dim))).shape}"
        )

        self.persistent_target_layer_net = th.nn.Sequential(
            th.nn.Linear(layer_dim, 16),
            th.nn.ReLU(),
        )

        self.expirable_target_layer_net = th.nn.Sequential(
            th.nn.Linear(layer_dim, 16),
            th.nn.ReLU(),
        )


        # self.target_layer_net = th.nn.Sequential(
        #     th.nn.Linear(layer_dim, 32),
        #     th.nn.ReLU(),
        #     th.nn.Linear(32, 16),
        #     th.nn.ReLU(),
        # )

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
            th.nn.Linear(
                n_flatten + 32 + 16 + 16 + 16, features_dim
            ),  # 32 for position features, 16 for layer features
            th.nn.ReLU(),
        )

    def forward(self, observations: dict) -> th.Tensor:
        device = next(self.parameters()).device

        # Process visual features (already in CHW format from CustomVecTranspose)
        # Note: we only convert to tensor once if not already a tensor
        if isinstance(observations["visual"], th.Tensor):
            visual_obs = observations["visual"].to(
                device, dtype=th.float32, non_blocking=True
            )
        else:
            visual_obs = th.as_tensor(
                observations["visual"], dtype=th.float32, device=device
            )

        # Add batch dimension (CHW -> NCHW)
        if visual_obs.dim() == 3:
            visual_obs = visual_obs.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

        # Apply resize function if needed
        if self.resize_to is not None:
            visual_obs = self.resize(visual_obs)

        visual_features = self.cnn(visual_obs)

        # Process position features
        if isinstance(observations["position"], th.Tensor):
            pos_obs = observations["position"].to(
                device, dtype=th.float32, non_blocking=True
            )
        else:
            pos_obs = th.as_tensor(
                observations["position"], dtype=th.float32, device=device
            )

        # Add batch dimension if needed
        if pos_obs.dim() == 1:
            pos_obs = pos_obs.unsqueeze(0)  # (2,) -> (1,2)

        pos_features = self.pos_net(pos_obs)

        # Process action history features
        # action_history_obs = th.as_tensor(observations["action_history"]).float()

        # Process layer features
        if isinstance(observations["current_layer"], th.Tensor):
            curr_layer_obs = observations["current_layer"].to(
                device, dtype=th.float32, non_blocking=True
            )
        else:
            curr_layer_obs = th.as_tensor(
                observations["current_layer"], dtype=th.float32, device=device
            )

        if isinstance(observations["persistent_target_layer"], th.Tensor):
            persistent_target_layer_obs = observations["persistent_target_layer"].to(
                device, dtype=th.float32, non_blocking=True
            )
        else:
            persistent_target_layer_obs = th.as_tensor(
                observations["persistent_target_layer"], dtype=th.float32, device=device
            )

        if isinstance(observations["expirable_target_layer"], th.Tensor):
            expirable_target_layer_obs = observations["expirable_target_layer"].to(
                device, dtype=th.float32, non_blocking=True
            )
        else:
            expirable_target_layer_obs = th.as_tensor(
                observations["expirable_target_layer"], dtype=th.float32, device=device
            )
        
        # if isinstance(observations["target_layer"], th.Tensor):
        #     target_layer_obs = observations["target_layer"].to(
        #         device, dtype=th.float32, non_blocking=True
        #     )
        # else:
        #     target_layer_obs = th.as_tensor(
        #         observations["target_layer"], dtype=th.float32, device=device
        #     )

        # Add batch dimension if needed
        # if action_history_obs.dim() == 1:
        #     action_history_obs = action_history_obs.unsqueeze(0)
        # action_history_features = self.action_history_net(action_history_obs)

        if curr_layer_obs.dim() == 1:
            curr_layer_obs = curr_layer_obs.unsqueeze(0)  # (1,) -> (1,1
        curr_layer_features = self.curr_layer_net(curr_layer_obs)

        if persistent_target_layer_obs.dim() == 1:
            persistent_target_layer_obs = persistent_target_layer_obs.unsqueeze(0)
        persistent_layer_features = self.persistent_target_layer_net(persistent_target_layer_obs)

        if expirable_target_layer_obs.dim() == 1:
            expirable_target_layer_obs = expirable_target_layer_obs.unsqueeze(0)
        expirable_layer_features = self.expirable_target_layer_net(expirable_target_layer_obs)

        # if target_layer_obs.dim() == 1:
        #     target_layer_obs = target_layer_obs.unsqueeze(0)  # (1,) -> (1,1
        # target_layer_features = self.target_layer_net(target_layer_obs)

        # Combine all features
        return self.combined(
            th.cat(
                [
                    visual_features,
                    pos_features,
                    curr_layer_features,
                    # target_layer_features,
                    persistent_layer_features,
                    expirable_layer_features,
                ],
                dim=1,
            )
        )


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
                "current_layer": venv.observation_space[
                    "current_layer"
                ],  # Keep layer space as is (already stacked)
                "persistent_target_layer": venv.observation_space[
                    "persistent_target_layer"
                ],
                "expirable_target_layer": venv.observation_space[
                    "expirable_target_layer"
                ],
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
            curr_layer = obs["current_layer"]  # Shape: (n_stack,)
            persistent_target_layer = obs["persistent_target_layer"]  # Shape: (1,)
            expirable_target_layer = obs["expirable_target_layer"]  # Shape: (1,)
        else:
            # For vectorized observations (e.g., from DummyVecEnv)
            # This is the most common case during training
            visual_obs = obs[0]["visual"]  # Shape: (H, W, C) or (1, H, W, C)
            position = obs[0]["position"]  # Shape: (2,) or (1, 2)
            curr_layer = obs[0]["current_layer"]  # Shape: (n_stack,) or (1, n_stack)
            persistent_target_layer = obs[0]["persistent_target_layer"]  # Shape: (1,) or (1, 1)
            expirable_target_layer = obs[0]["expirable_target_layer"]  # Shape: (1,) or (1, 1)

        # Single fast check for both visual and position to remove batch dim
        # Remove any extra batch dimension from DummyVecEnv if present
        if isinstance(visual_obs, np.ndarray) and len(visual_obs.shape) == 4:
            visual_obs = visual_obs[0]  # Convert (1, H, W, C) to (H, W, C)

        if isinstance(position, np.ndarray) and len(position.shape) == 2:
            position = position[0]  # Convert (1, 2) to (2,)

        if isinstance(curr_layer, np.ndarray):
            if len(curr_layer.shape) > 1:
                curr_layer = curr_layer[0]  # Convert (1, n_stack) to (n_stack,)
            if len(curr_layer.shape) == 0:
                curr_layer = curr_layer.reshape(1)  # Ensure shape is (1,)

        if isinstance(persistent_target_layer, np.ndarray):
            if len(persistent_target_layer.shape) > 1:
                persistent_target_layer = persistent_target_layer[0]  # Convert (1, 1) to (1,)
            if len(persistent_target_layer.shape) == 0:
                persistent_target_layer = persistent_target_layer.reshape(1)  # Ensure shape is (1,)

        if isinstance(expirable_target_layer, np.ndarray):
            if len(expirable_target_layer.shape) > 1:
                expirable_target_layer = expirable_target_layer[0]  # Convert (1, 1) to (1,)
            if len(expirable_target_layer.shape) == 0:
                expirable_target_layer = expirable_target_layer.reshape(1)  # Ensure shape is (1,)

        # Convert from HWC to CHW format - only once and only if needed
        # Check if already in CHW format to avoid unnecessary transpose
        if (
            isinstance(visual_obs, np.ndarray)
            and len(visual_obs.shape) == 3
            and visual_obs.shape[2] <= 12
        ):
            # Use numpy's optimized transpose instead of manual loop
            visual_result = np.ascontiguousarray(np.transpose(visual_obs, (2, 0, 1)))
        else:
            # Already in the right format or special case, no need to transpose
            visual_result = visual_obs

        return {
            "visual": visual_result,
            "position": position,
            "current_layer": curr_layer,
            "persistent_target_layer": persistent_target_layer,
            "expirable_target_layer": expirable_target_layer,
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
    env = TimeLimit(env, max_episode_steps=100)
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
        self.last_memory_check = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_memory_check > 50000:
            self.last_memory_check = self.num_timesteps
            gc.collect()
            if th.cuda.is_available():
                th.cuda.empty_cache()

            # Log memory usage if verbose
            if self.verbose > 0:
                import psutil

                memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
                gpu_memory = ""
                if th.cuda.is_available():
                    gpu_memory = f", GPU memory: {th.cuda.memory_allocated() / 1024 / 1024:.1f}MB"
                print(
                    f"Step {self.num_timesteps}: Memory usage: {memory_usage:.2f}MB{gpu_memory}"
                )

        return True


# Only run training code when script is executed directly (not imported)
if __name__ == "__main__":
    # Training parameters
    total_timesteps = 5000000
    check_freq = 100000  # Save checkpoint every 50k steps
    MODEL_TAG = "PPO_expirable_target_nn_Continuous_v2"

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
            n_stack=n_stack,
        ),
        # Use a larger network to process the stacked frames
        net_arch=[256, 128, 64],
        # net_arch=[256, 256],
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
        target_kl=0.05,
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

    # Save the fdatetimeel
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"ppo_visual_attention_full_{timestamp}")
    print(f"Final model saved as ppo_visual_attention_full_{timestamp}")
    print("Model training complete!")
    print("To properly evaluate the model with human rendering, use:")
    print(f"python evaluate_model.py ppo_visual_attention_full_{timestamp}")
    print(
        "This will handle the observation stacking and preprocessing correctly for visual rendering."
    )
