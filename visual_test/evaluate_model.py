import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
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
    env = Monitor(env)
    return env


def manual_evaluate(model, env, n_episodes=10):
    """
    Manually evaluates a model with frame stacking.
    Handles observation preprocessing to match training format.
    """
    rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        step_count = 0

        # Initialize frame stacking with duplicate frames
        frames_history = [obs["visual"]] * STACKED_FRAMES
        positions_history = [obs["position"]] * STACKED_FRAMES
        layers_history = [obs["current_layer"]] * STACKED_FRAMES

        while not (done or truncated) and step_count < 100:
            try:
                # Stack frames and convert to CHW format
                stacked_frames = []
                for frame in frames_history:
                    channels_first = np.transpose(frame, (2, 0, 1))
                    stacked_frames.append(channels_first)

                stacked_visual = np.concatenate(stacked_frames, axis=0)
                stacked_position = np.concatenate(positions_history)
                stacked_layers = np.concatenate(layers_history)

                # Ensure correct data types
                if stacked_visual.dtype != np.uint8:
                    stacked_visual = stacked_visual.astype(np.uint8)
                if stacked_position.dtype != np.float32:
                    stacked_position = stacked_position.astype(np.float32)
                if stacked_layers.dtype != np.int8:
                    stacked_layers = stacked_layers.astype(np.int8)

                stacked_obs = {
                    "visual": stacked_visual,
                    "position": stacked_position,
                    "current_layer": stacked_layers,
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

                # Update frame history
                frames_history.pop(0)
                frames_history.append(next_obs["visual"])
                positions_history.pop(0)
                positions_history.append(next_obs["position"])
                layers_history.pop(0)
                layers_history.append(next_obs["current_layer"])

                # Render and add delay for visibility
                env.render()
                pygame.time.wait(50)

            except Exception as e:
                print(f"Error during episode {episode+1}, step {step_count}: {e}")
                import traceback

                traceback.print_exc()
                break

        rewards.append(episode_reward)
        print(f"Episode {episode+1} reward: {episode_reward}, steps: {step_count}")

    mean_reward = sum(rewards) / len(rewards) if rewards else 0
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
        print(f"Mean reward: {mean_reward:.2f}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()

    # Clean up
    eval_env.close()
