import gymnasium as gym
import numpy as np

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

def buildEnv():
    """
    Build the Tetris environment with specific wrappers for grouped actions and feature vector observations.
    
    Returns:
        env (gym.Env): The configured Tetris environment.
    """
    # Create the Tetris environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human")
    
    # Wrap the environment to group actions and observations
    env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)])
    
    return env

env = buildEnv()
obs, info = env.reset(seed=42)

while True:

    # Print detailed observation space information
    print("\nObservation Space Details:")
    print(f"Observation Space Type: {type(env.observation_space)}")
    print(f"Observation Space Shape: {env.observation_space.shape}")
    print(f"Observation Space Bounds:")
    print(f"- Low: {env.observation_space.low}")
    print(f"- High: {env.observation_space.high}")

    # Print actual observation example
    print("\nExample Observation:")
    print(f"Observation Shape: {obs.shape}")
    print(f"Observation Content: {obs}")
    print(f"Observation Type: {type(obs)}")

    action = env.action_space.sample()
    # Processing
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

print("Environment built successfully!")

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)