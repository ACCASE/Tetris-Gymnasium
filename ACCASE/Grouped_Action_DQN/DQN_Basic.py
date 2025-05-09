import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import cv2
import yaml
import matplotlib
import matplotlib.pyplot as plt
import os
import itertools
import datetime
import argparse

from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

######## Saving and Storing Logs and Models #########
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" # Date
RUNS_DIR = "runs/DQN_Basic" # Directory to save runs
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg'
matplotlib.use('Agg') 

# Choose device. Look for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
def buildEnv(render=False):
    """
    Build the Tetris environment with specific wrappers for grouped actions and feature vector observations.
    
    Returns:
        env (gym.Env): The configured Tetris environment.
    """
    # Create the Tetris environment
    env = gym.make("tetris_gymnasium/Tetris", render_mode="human" if render else None)
    
    # Wrap the environment to group actions and observations
    env = GroupedActionsObservations(
                env, observation_wrappers=[FeatureVectorObservation(env)])
    
    return env

class Agent:
    def __init__(self, params_set):
        with open('ACCASE/Grouped_Action_DQN/params.yml', 'r') as file:
            all_params_sets = yaml.safe_load(file)
            params = all_params_sets[params_set]

        self.hidden_layer_dim = params['hidden_layer_dim']
        self.num_episodes = params['num_episodes']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']  # Discount factor for future rewards

        self.loss_fn = nn.MSELoss()  # Loss function for Q-learning
        self.optimizer = None  # Optimizer for training

        # Visualzation and Storage
        self.LOG_FILE   = os.path.join(RUNS_DIR, f"{params_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{params_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{params_set}.png")

    def optimize(self, reward, action, state, new_state, terminated, network):
        """
        Optimize the DQN using the Bellman equation
        
        Args:
            expected_reward (torch.Tensor): Expected reward
        """

        # Get the expected reward for the current state and action
        current_q = network(state)[action]

        # Compute target Q-value using Bellman equation
        with torch.no_grad():
            # Get max Q-value for next state
            next_q_values = network(new_state)
            next_q_value = next_q_values.max()
        
        # Calculate the target reward/update
        if terminated:
            target_q = reward
        else:
            target_q = reward + self.discount_factor * next_q_value

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_graph(self, rewards_list, epsilon_history):
        """
        Save the graph of rewards and epsilon history
        """
        fig = plt.figure(1)

        # Plot rewards
        window_size = 10
        rolling_avg_reward = np.convolve(rewards_list, np.ones(window_size)/window_size, mode='valid')

        # Different mean reward calc
        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_list))
        for i in range(len(mean_rewards)):
            mean_rewards[i] = np.mean(rewards_list[max(0, i-99):(i+1)])

        # Plot Norm^2 Error vs episodes
        plt.figure(1,figsize=(10,5))
        plt.subplot(1,2,1)
        plt.plot(rewards_list, label='Rewards')
        plt.plot(np.arange(window_size/2 - 1, len(rewards_list)-window_size/2), rolling_avg_reward, label='Rolling Average')
        plt.plot(mean_rewards, label='Mean Rewards') 
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Rewards vs Episodes')

        # Plot Epsilon history
        plt.subplot(1,2,2)
        plt.plot(epsilon_history, label='Epsilon')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Epsilon vs Episodes')

        plt.subplots_adjust(wspace=1, hspace=1)

        # Save Figures
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def normalize_state(self, state, num_rows, num_cols):
        """
        Normalize the state to be between 0 and 1
        """
        # Normalize the state by dividing by the maximum value in the state
        # Build Vector to normalize state by
        # Column Height and Max Height
        normalization_vector = torch.ones(1,num_cols+1, device=device) * num_rows
        # Holes
        max_holes = num_cols*num_rows
        normalization_vector = torch.cat((normalization_vector, torch.ones(1,1, device=device) * max_holes), dim=1)
        # Bumpiness
        max_bumpiness = num_rows
        normalization_vector = torch.cat((normalization_vector, torch.ones(1,1, device=device) * max_bumpiness), dim=1)
        # Normalize the state
        normalized_state = state / normalization_vector
        return normalized_state

    def run(self, training=True, render=False):
        """
        Train the agent using Deep Q-Learning
        """
        env = buildEnv(render)
        
        # Storage
        rewards_list = []
        epsilon_history = []
        lines_cleared_list = []

        # Get input and output dimensions
        num_states = env.observation_space.shape[1]
        num_actions = 1 #env.action_space.n
        num_rows = env.unwrapped.height
        num_cols = env.unwrapped.width
        
        # Initialize DQN
        policy_net = DQN(num_states, num_actions).to(device)

        # Load model if not training
        if not training:
            policy_net.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_net.eval()

        # Training parameters
        if training:
            epsilon = self.epsilon_start
            epsilon_decay = self.epsilon_decay

            # Initialize optimizer
            self.optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

            # Start time
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time

            # Storage
            best_reward = float('-inf')
        
        for episode in itertools.count():
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            # state = self.normalize_state(state, num_rows, num_cols) # Normalize state
            env.render()
            terminated = False
            truncated = False
            total_reward = 0
            episode_lines_cleared = 0
            
            while not terminated and not truncated:

                # Get action mask from info
                action_mask = torch.tensor(info['action_mask'], dtype=torch.bool, device=device)  # Get valid actions
                
                if training and random.random() < epsilon:
                    # Choose random action
                    # action = env.action_space.sample()
                    # action = torch.tensor(action, dtype=torch.int, device=device)

                    # Get valid action indices
                    valid_actions = torch.where(action_mask)[0]
                    # Sample random action from valid actions only
                    action = valid_actions[torch.randint(0, len(valid_actions), (1,))].item()
                    action = torch.tensor(action, dtype=torch.int, device=device)

                else:
                    # Choose action based on policy network
                    # Create a batch of all possible next states
                    
                    # possible_states = state.repeat(env.action_space.n, 1)  # Repeat state for each action
                    
                    # Get Q-values for all possible actions
                    with torch.no_grad():  # No need to track gradients for prediction
                        q_values = policy_net(state)  # Add batch dimension
                    
                    # Mask invalid actions with large negative value
                    q_values[~action_mask] = float('-inf')
                    
                    # Choose action with highest Q-value
                    action = q_values.argmax().item()
                    action = torch.tensor(action, dtype=torch.int, device=device)

                key = cv2.waitKey(1) # Needed to render the environment for some reason

                # Step to next state with action
                new_state, reward, terminated, truncated, info = env.step(action.item()) # .item() returns tensor value
                total_reward += reward
                episode_lines_cleared = info['lines_cleared']

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                # new_state = self.normalize_state(new_state, num_rows, num_cols) # Normalize state
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if training:
                    self.optimize(reward, action, state, new_state, terminated, policy_net)

                # Next state
                state = new_state

                # Optionally Render the enviornment
                env.render()

            # Add reward to list
            rewards_list.append(total_reward)
            lines_cleared_list.append(episode_lines_cleared)
        
            if training:
                epsilon_history.append(epsilon)

                # Save Logs
                if total_reward > best_reward:
                    # Create log message
                    log = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {total_reward} at episode {episode}"
                    print(log) # print log to console
                    with open(self.LOG_FILE, 'a') as log_file:
                        log_file.write(log + "\n") # Save log file
                    # Update new best reward
                    best_reward = total_reward 
                    torch.save(policy_net.state_dict(), self.MODEL_FILE) # Save model

                # Save Graph
                current_time = datetime.datetime.now()
                if current_time-last_graph_update_time > datetime.timedelta(seconds=10):
                    self.save_graph(rewards_list, epsilon_history)
                    last_graph_update_time = current_time
                    
            if training:
                # Modify epsilon
                epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

            # print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        
        env.close()
        return rewards_list

if __name__ == "__main__":

        # Add option for command line arguments
    parser = argparse.ArgumentParser(description='Training or Testing DQN Agent')
    parser.add_argument('params', type=str, help='Parameters set to use for training/testing')
    parser.add_argument('--train', action='store_true', help='Training flag')
    args = parser.parse_args()

    terryTetris = Agent(params_set=args.params)

    if args.train:
        # Training Mode
        terryTetris.run(training=True, render=False)
    else:
        # Testing Mode
        terryTetris.run(training=False, render=True)
    # paul = Agent('Tetris1')
    # rewards = paul.run(True, False)

    # Plotting

    # window_size = 10
    # rolling_avg_reward = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    # # Plot Norm^2 Error vs episodes
    # plt.figure(1,figsize=(10,5))
    # # plt.subplot(1,2,1)
    # plt.plot(rewards, label='Rewards')
    # plt.plot(np.arange(window_size/2 - 1, len(rewards)-window_size/2), rolling_avg_reward, label='Rolling Average') 
    # plt.xlabel('Episode')
    # plt.ylabel('Total Reward')
    # plt.title('Rewards vs Episodes')
    # # plt.subplot(1,2,2)
    # # plt.plot(Qe_error, label='e Greedy Q-Learning')
    # # plt.xlabel('Episode')
    # # plt.ylabel('E[||Qk-Q*||]^2')
    # # plt.title('epsilon-Greedy Q-Learning Performance ($epsilon$ = ' + str(epsilon) + ')')
    # plt.show()