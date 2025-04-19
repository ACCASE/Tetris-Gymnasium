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

######## Saving and Storing Logs and Models #########
DATE_FORMAT = "%Y-%m-%d %H:%M:%S" # Date
RUNS_DIR = "runs/DQN_Replay" # Directory to save runs
os.makedirs(RUNS_DIR, exist_ok=True)

# 'Agg'
matplotlib.use('Agg') 


# Choose device. Look for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu' # Uncomment this line to force CPU usage

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
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

class ReplayBuffer():
    def __init__(self, capacity):
        self.buffer = deque([],maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, params_set):
        with open('ACCASE/Grouped_Action_DQN/params.yml', 'r') as file:
            all_params_sets = yaml.safe_load(file)
            params = all_params_sets[params_set]

        self.hidden_layer_dim = params['hidden_layer_dim'] # Hidden layer dimension
        self.num_episodes = params['num_episodes']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_end = params['epsilon_end']
        self.epsilon_decay = params['epsilon_decay']
        self.learning_rate = params['learning_rate']
        self.discount_factor = params['discount_factor']  # Discount factor for future rewards
        self.replay_buffer_size = params['replay_buffer_size']  # Size of the replay buffer
        self.batch_size = params['batch_size']  # Batch size for training
        self.target_update_freq = params['target_update_freq']  # Frequency of target network updates

        # Network Elements
        self.loss_fn = nn.MSELoss()  # Loss function for Q-learning
        self.optimizer = None  # Optimizer for training

        # Visualzation and Storage
        self.LOG_FILE   = os.path.join(RUNS_DIR, f"{params_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{params_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{params_set}.png")

    def optimize(self, batch, policy_net, target_net):
        """
        Optimize the DQN
        """
        # Separate list into each element
        states, actions, rewards, new_states, terminations, action_masks = zip(*batch) # *List is transposed
        # Action mask is inverted i.e. false is true and true is false

        # Convert to batches of tensors
        states = torch.stack(states).to(device) # Stack tensors into a batch
        actions = torch.stack(actions).long().to(device) # Stack tensors into a batch
        rewards = torch.stack(rewards).to(device) # Stack tensors into a batch
        new_states = torch.stack(new_states).to(device) # Stack tensors into a batch
        terminations = torch.tensor(terminations).float().to(device) # Stack tensors into a batch
        action_masks = (torch.stack(action_masks)).float().to(device) # Convert to tensor with 1s and 0s

        with torch.no_grad():  # No need to track gradients for prediction
            # action_masks = torch.tensor(infos['action_mask'], dtype=torch.bool, device=device) # Limit to valid actions
            # new_state_q = target_net(new_state) # Get Q-values for all possible actions
            # new_state_q[~action_mask] = float('-inf') # Mask invalid actions with large negative value
            target_q = rewards + (1-terminations) * self.discount_factor * (target_net(new_states).squeeze(2)*action_masks).max(1)[0] # Bellman equation

        # Get the expected reward for the current state and action
        current_q = policy_net(states).squeeze(2).gather(dim=1, index=actions.unsqueeze(1)).squeeze() # Get Q-values for all possible actions
        # current_q = policy_net(state)[action]

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad() # Reset gradients
        loss.backward() # Backpropagation
        self.optimizer.step() # Update weights

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

    def run(self, training=True, render=False):
        """
        Train the agent using Deep Q-Learning
        """
        env = buildEnv(render)
        
        # Storage
        rewards_list = []
        epsilon_history = []
        best_reward = -float('inf')

        # Get input and output dimensions
        num_states = env.observation_space.shape[1]
        num_actions = 1 #env.action_space.n Becuase of grouped actions
            # Will evaluate the value of each action and choose the greatest
        
        # Initialize DQN
        policy_net = DQN(num_states, num_actions, self.hidden_layer_dim).to(device)

        # Training parameters
        if training:
            # Initialize target network
            target_net = DQN(num_states, num_actions).to(device)
            target_net.load_state_dict(policy_net.state_dict())
            # Track Steps. Every so often update the target network with policy
            step_count = 0
            start_time = datetime.datetime.now()
            last_graph_update_time = start_time


            # Initialize epsilon and decay
            epsilon = self.epsilon_start
            epsilon_decay = self.epsilon_decay

            # Initialize replay buffer
            replay_buffer = ReplayBuffer(self.replay_buffer_size)

            # Initialize optimizer. Currlently using Adam optimizer
            self.optimizer = optim.Adam(policy_net.parameters(), lr=self.learning_rate)
        
        for episode in itertools.count():
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            env.render()
            total_reward = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:

                # Get action mask from info
                action_mask = torch.tensor(info['action_mask'], dtype=torch.bool, device=device)  # Get valid actions
                
                if training and random.random() < epsilon:
                    # Get valid action indices
                    valid_actions = torch.where(action_mask)[0]
                    # Sample random action from valid actions only
                    action = valid_actions[torch.randint(0, len(valid_actions), (1,))].item()
                    action = torch.tensor(action, dtype=torch.int, device=device)
                else:
                    # Choose action based on policy network
                    # Have a batch of all possible next states                    
                    # Get Q-values for all possible actions
                    with torch.no_grad():  # No need to track gradients for prediction
                        q_values = policy_net(state)  # Add batch dimension
                    
                    # Mask invalid actions with large negative value
                    q_values[~action_mask] = float('-inf')
                    
                    # Choose action with highest Q-value
                    action = q_values.argmax().item()
                    action = torch.tensor(action, dtype=torch.int, device=device)

                # key = cv2.waitKey(1) # Needed to render the environment for some reason

                # Step to next state with action
                new_state, reward, terminated, truncated, info = env.step(action.item()) # .item() returns tensor value
                total_reward += reward

                # Convert new state and reward to tensors
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                

                if training:
                    # Store experience in replay buffer
                    replay_buffer.add((state, action, reward, new_state, terminated, action_mask))

                    # Count for updating target network
                    step_count += 1

                # Next state
                state = new_state

                # Optionally Render the enviornment
                env.render()

            # Add reward to list
            rewards_list.append(total_reward)
            

            # Save model when a new best reward is achieved
            if training:
                epsilon_history.append(epsilon)

                # Modify epsilon
                epsilon = max(self.epsilon_end, epsilon * self.epsilon_decay)

                # If enough samples have been collected, sample a batch from the replay buffer
                if len(replay_buffer) > self.batch_size:
                    batch = replay_buffer.sample(32)
                    self.optimize(batch, policy_net, target_net)

                    # Sync target network with policy network every so often
                    if step_count > self.target_update_freq:
                        target_net.load_state_dict(policy_net.state_dict())
                        step_count = 0 # reset step count

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