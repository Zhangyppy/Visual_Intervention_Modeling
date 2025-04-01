import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.wrappers import TimeLimit
import torch as th

# Import the model and environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

# Import the environment class from the main file
from visual_gazepoint_to_notification import VisualSimulationEnv

# Constants from the original file
SCREEN_WIDTH = 200
SCREEN_HEIGHT = 200
AGENT_RADIUS = 5
MAX_MOVE_SPEED = 20
MIN_MOVE_SPEED = 5
ENVIRONMENT_BLOCK_SIZE = 20
NOTIFICATION_BLOCK_SIZE = 40
STACKED_FRAMES = 4


def create_simple_eval_env(render_mode="human"):
    """Creates a simple environment for human rendering without wrappers."""
    env = VisualSimulationEnv(render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=100)
    env = Monitor(env)
    return env


def manual_evaluate(model, env, n_episodes=10):
    """
    Manually evaluates a model with frame stacking.
    Handles observation preprocessing to match training format.
    """
    rewards = []
    targets_collected = []
    persistent_visits = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0
        episode_targets = 0
        persistent_target_visits = 0

        # Initialize frame stacking with duplicate frames
        frames_history = [obs["visual"]] * STACKED_FRAMES
        positions_history = [obs["position"]] * STACKED_FRAMES
        curr_layer_history = [obs["current_layer"]] * STACKED_FRAMES
        # target_layer_history = [obs["target_layer"]] * STACKED_FRAMES
        persistent_target_layer_history = [obs["persistent_target_layer"]] * STACKED_FRAMES
        expirable_target_layer_history = [obs["expirable_target_layer"]] * STACKED_FRAMES

        # Track previous target completion state to detect new completions
        prev_target_completed = False

        while not (done or truncated):
            try:
                # Stack frames and convert to CHW format
                stacked_frames = []
                for frame in frames_history:
                    channels_first = np.transpose(frame, (2, 0, 1))
                    stacked_frames.append(channels_first)

                stacked_visual = np.concatenate(stacked_frames, axis=0)
                stacked_position = np.concatenate(positions_history)
                stacked_current_layer = np.concatenate(curr_layer_history)
                # stacked_target_layer = np.concatenate(target_layer_history)
                stacked_persistent_target_layer = np.concatenate(
                    persistent_target_layer_history
                )
                stacked_expirable_target_layer = np.concatenate(
                    expirable_target_layer_history
                )

                # Ensure correct data types
                if stacked_visual.dtype != np.uint8:
                    stacked_visual = stacked_visual.astype(np.uint8)
                if stacked_position.dtype != np.float32:
                    stacked_position = stacked_position.astype(np.float32)
                if stacked_current_layer.dtype != np.int8:
                    stacked_current_layer = stacked_current_layer.astype(np.int8)
                # if stacked_target_layer.dtype != np.int8:
                #     stacked_target_layer = stacked_target_layer.astype(np.int8)
                if stacked_persistent_target_layer.dtype != np.int8:
                    stacked_persistent_target_layer = stacked_persistent_target_layer.astype(np.int8)
                if stacked_expirable_target_layer.dtype != np.int8:
                    stacked_expirable_target_layer = stacked_expirable_target_layer.astype(np.int8)

                stacked_obs = {
                    "visual": stacked_visual,
                    "position": stacked_position,
                    "current_layer": stacked_current_layer,
                    # "target_layer": stacked_target_layer,
                    "persistent_target_layer": stacked_persistent_target_layer,
                    "expirable_target_layer": stacked_expirable_target_layer,
                }

                # Get and convert action
                action, _ = model.predict(stacked_obs, deterministic=True)
                # NOTE: With continuous actions (Box space with shape (2,)),
                # we need to keep the action as a numpy array of shape (2,)
                # representing movement in [x, y] directions

                # Handle both movement and layer actions if present
                if isinstance(action, tuple) and len(action) == 2:
                    movement_action, layer_action = action
                else:
                    movement_action = action
                    layer_action = None

                # Execute action
                next_obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Check if targets were collected
                # Look at the environment's current target completion status
                if (
                    hasattr(env, "target")
                    and hasattr(env, "has_expirable_target")
                    and env.has_expirable_target
                    and env.target.completed
                    and not prev_target_completed
                ):
                    episode_targets += 1
                    prev_target_completed = True
                elif (
                    hasattr(env, "target")
                    and hasattr(env, "has_expirable_target")
                    and env.has_expirable_target
                    and not env.target.completed
                ):
                    prev_target_completed = False

                # Track visits to persistent target
                if hasattr(env, "persistent_target") and hasattr(env, "current_layer"):
                    if (
                        env.persistent_target.check_collision(
                            env.agent_pos, AGENT_RADIUS
                        )
                        and env.current_layer == env.persistent_target.layer
                    ):
                        persistent_target_visits += 1

                # Update frame history
                frames_history.pop(0)
                frames_history.append(next_obs["visual"])
                positions_history.pop(0)
                positions_history.append(next_obs["position"])
                curr_layer_history.pop(0)
                curr_layer_history.append(next_obs["current_layer"])
                # target_layer_history.pop(0)
                # target_layer_history.append(next_obs["target_layer"])

                # Render and add delay for visibility
                env.render()
                pygame.time.wait(50)

            except Exception as e:
                print(f"Error during episode {episode+1}, step {step_count}: {e}")
                import traceback

                traceback.print_exc()
                break

        rewards.append(episode_reward)
        targets_collected.append(episode_targets)
        persistent_visits.append(persistent_target_visits)

        print(f"Episode {episode+1}:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Expirable targets collected: {episode_targets}")
        print(f"  Persistent target interactions: {persistent_target_visits}")
        print("-------------------------------")

    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    mean_targets = (
        sum(targets_collected) / len(targets_collected) if targets_collected else 0
    )
    mean_persistent = (
        sum(persistent_visits) / len(persistent_visits) if persistent_visits else 0
    )

    print("\nEvaluation Summary:")
    print(f"  Mean reward: {mean_reward:.2f}")
    print(f"  Mean expirable targets collected: {mean_targets:.2f}")
    print(f"  Mean persistent target interactions: {mean_persistent:.2f}")

    return mean_reward


if __name__ == "__main__":
    # Get model path from command line or find most recent
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_files = [
            f for f in os.listdir(".") if f.startswith("ppo_visual_attention_full_")
        ]
        if not model_files:
            print("No model files found. Please specify a model path.")
            sys.exit(1)
        model_path = max(model_files)

    print(f"Loading model from {model_path}")

    # Load model and create environment
    model = PPO.load(model_path)
    print("Creating evaluation environment...")
    eval_env = create_simple_eval_env(render_mode="human")

    # Evaluate model
    print("Evaluating model...")
    try:
        mean_reward = manual_evaluate(model, eval_env, n_episodes=10)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()

    # Clean up
    eval_env.close()
