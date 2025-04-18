import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32):
        """
        Initialize Deep Q-Network
        
        Args:
            input_dim (int): Dimension of state space
            output_dim (int): Dimension of action space
        """
        super(DQN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
    
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

def train():
    """
    Train the agent using Deep Q-Learning
    """
    env = buildEnv()
    
    # Get input and output dimensions
    state_dim = env.observation_space.shape[1]
    action_dim = env.action_space.n
    
    # Initialize DQN and target network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # Setup training parameters
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    memory = deque(maxlen=10000)  # Experience replay buffer
    batch_size = 64
    gamma = 0.99  # Discount factor
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    target_update = 10  # How often to update target network
    
    for episode in range(1000):
        obs, info = env.reset()
        total_reward = 0
        state = obs
        
        while True:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(state_tensor).max(1)[1].item()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            # Store transition in memory
            memory.append((state, action, reward, next_state, terminated))
            
            # Train on random batch from memory
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                # Convert to tensors
                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device)
                
                # Compute current Q values
                current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
                
                # Compute next Q values
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    next_q_values[dones] = 0.0
                    
                # Compute target Q values
                target_q_values = rewards + gamma * next_q_values
                
                # Compute loss and update
                loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            state = next_state
            
            if terminated or truncated:
                break
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if episode % 10 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
    
    env.close()
    return policy_net


train()
network = DQN(13, 1)  # Initialize the DQN (Input state size, Output action size)