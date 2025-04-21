import datetime
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
import itertools
import os
import datetime
import argparse

from collections import deque
from tetris_gymnasium.envs.tetris import Tetris
from tetris_gymnasium.wrappers.grouped import GroupedActionsObservations
from tetris_gymnasium.wrappers.observation import FeatureVectorObservation

##############
## Build a cost function for the Tetris game
## Manually decide weights for each feature
##############


######## Saving and Storing Logs and Models #########
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" # Date
RUNS_DIR = "runs/Hand_Crafted" # Directory to save runs
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg'
matplotlib.use('Agg') 

# Choose device. Look for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # Uncomment this line to force CPU usage
    
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
        with open('ACCASE/Grouped_Action_Hand_Crafted/params.yml', 'r') as file:
            all_params_sets = yaml.safe_load(file)
            params = all_params_sets[params_set]

        self.column0_height = params['column0_height']
        self.column1_height = params['column1_height']
        self.column2_height = params['column2_height']
        self.column3_height = params['column3_height']
        self.column4_height = params['column4_height']
        self.column5_height = params['column5_height']
        self.column6_height = params['column6_height']
        self.column7_height = params['column7_height']
        self.column8_height = params['column8_height']
        self.column9_height = params['column9_height']
        self.max_height = params['max_height']
        self.holes = params['holes']
        self.bumpiness = params['bumpiness']

        self.weights = torch.tensor([
            self.column0_height,
            self.column1_height,
            self.column2_height,
            self.column3_height,
            self.column4_height,
            self.column5_height,
            self.column6_height,
            self.column7_height,
            self.column8_height,
            self.column9_height,
            self.max_height,
            self.holes,
            self.bumpiness
        ], dtype=torch.float, device=device)

        # Visualzation and Storage
        self.LOG_FILE   = os.path.join(RUNS_DIR, f"{params_set}.log")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{params_set}.png")

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

    def cost_function(self, state):
        """
        Cost function to evaluate the state of the Tetris game.
        """
        # Multiply the batch of states with the weights
        weighted_state = state * self.weights.unsqueeze(0)
        costs = torch.sum(weighted_state, dim=1)  # Sum across the features
        return costs
    
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
        best_reward = -float('inf')

        # Get input and output dimensions
        num_states = env.observation_space.shape[1]
        num_actions = 1 #env.action_space.n Becuase of grouped actions
            # Will evaluate the value of each action and choose the greatest
        num_rows = env.unwrapped.height
        num_cols = env.unwrapped.width

        # Track Steps. Every so often update the target network with policy
        step_count = 0
        start_time = datetime.datetime.now()
        last_graph_update_time = start_time
    
        for episode in itertools.count():
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            state = self.normalize_state(state, num_rows, num_cols) # Normalize state
            env.render()
            total_reward = 0
            terminated = False
            truncated = False

            # Stats
            episode_lines_cleared = 0
            episode_length = 0
            
            while not terminated and not truncated:

                # Get action mask from info
                action_mask = torch.tensor(info['action_mask'], dtype=torch.bool, device=device)  # Get valid actions

                costs = self.cost_function(state)
                action = torch.argmin(costs[action_mask == 1])
                # q_values[:, action_mask == 1, :] = policy_net(
                #     torch.Tensor(state[action_mask == 1, :]).to(device)
                # )
                # action = torch.argmax(q_values, dim=1)[0]

                # # Choose action based on policy network
                # # Have a batch of all possible next states                    
                # # Get Q-values for all possible actions
                # with torch.no_grad():  # No need to track gradients for prediction
                #     q_values = policy_net(state)  # Add batch dimension
                
                # # Mask invalid actions with large negative value
                # q_values[~action_mask] = float('-inf')
                
                # # Choose action with highest Q-value
                # action = q_values.argmax().item()
                # action = torch.tensor(action, dtype=torch.int, device=device)

                key = cv2.waitKey(1) # Needed to render the environment for some reason

                # Step to next state with action
                new_state, reward, terminated, truncated, info = env.step(action.item()) # .item() returns tensor value
                total_reward += reward
                episode_lines_cleared += info['lines_cleared']
                episode_length += 1

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                new_state = self.normalize_state(new_state, num_rows, num_cols) # Normalize state
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                


                # Next state
                state = new_state

                # Optionally Render the enviornment
                env.render()

            # Add reward to list
            rewards_list.append(total_reward)
            lines_cleared_list.append(episode_lines_cleared)
            

            # Save Logs
            if total_reward > best_reward:
                # Create log message
                log = f"{datetime.datetime.now().strftime(DATE_FORMAT)}: New best reward: {total_reward} at episode {episode}. Cleard lines: {episode_lines_cleared}"
                print(log) # print log to console
                with open(self.LOG_FILE, 'a') as log_file:
                    log_file.write(log + "\n") # Save log file
                # Update new best reward
                best_reward = total_reward 

            # Save Graph
            current_time = datetime.datetime.now()
            if current_time-last_graph_update_time > datetime.timedelta(seconds=10):
                self.save_graph(rewards_list, epsilon_history)
                last_graph_update_time = current_time

            if not training:
                print(
                    f"episode={episode}, "
                    f"episodic_return={total_reward}, "
                    f"episodic_length={episode_length}, "
                    f"lines_cleared={episode_lines_cleared}"
                )
        # env.close()
        # return rewards_list

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