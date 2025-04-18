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
    env = GroupedActionsObservations(env)
    
    # Wrap the environment to use feature vector observations
    env = FeatureVectorObservation(env)
    
    return env

def train():
    """
    Train the agent in the Tetris environment.
    
    Returns:
        None
    """
    # Initialize the environment
    env = buildEnv()
    
    # Reset the environment to start a new episode
    obs, info = env.reset()
    
    # Example training loop (to be replaced with actual training logic)
    for episode in range(1000):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, terminated, truncated, info = env.step(action)  # Take a step in the environment
        
        if terminated or truncated:
            obs, info = env.reset()  # Reset the environment if the episode is done


    # Close the environment after training
    env.close()

train()

# env = buildEnv()
print("Environment built successfully!")

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)