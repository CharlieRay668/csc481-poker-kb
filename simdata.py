import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import rlcard
# from rlcard.agents import RandomAgent
# from rlcard.utils import tournament
import csv

# Set up environment
env = rlcard.make('leduc-holdem')

# Parameters
REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
LR = 0.0001
EPSILON = 1.0
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1
EPISODES = 10000
TARGET_UPDATE = 100
DATASET_PATH = "leduc_dataset.csv"

# Create dataset file
with open(DATASET_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SLNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(SLNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

class NFSPAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_network = DQN(state_size, action_size)
        self.target_q_network = DQN(state_size, action_size)
        self.sl_network = SLNetwork(state_size, action_size)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=LR)
        self.sl_optimizer = optim.Adam(self.sl_network.parameters(), lr=LR)

        self.q_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.sl_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.epsilon = EPSILON

    def select_action(self, state, use_sl=False):
        state = torch.FloatTensor(state)
        if use_sl or random.random() > self.epsilon:
            with torch.no_grad():
                action_probs = self.sl_network(state)
                action = torch.argmax(action_probs).item()
        else:
            action = random.randint(0, self.action_size - 1)
        return action

    def store_experience(self, state, action, reward, next_state, done):
        self.q_memory.append((state, action, reward, next_state, done))
        self.sl_memory.append((state, action))
        # Append to dataset
        with open(DATASET_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([state, action, reward, next_state, done])

    def update_q_network(self):
        if len(self.q_memory) < BATCH_SIZE:
            return

        batch = random.sample(self.q_memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q_values = self.target_q_network(next_states).max(1)[0]
        target_q_values = rewards + (GAMMA * next_q_values * (1 - dones))

        loss = nn.MSELoss()(q_values, target_q_values.detach())
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def update_sl_network(self):
        if len(self.sl_memory) < BATCH_SIZE:
            return

        batch = random.sample(self.sl_memory, BATCH_SIZE)
        states, actions = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)

        predicted_actions = self.sl_network(states)
        loss = nn.CrossEntropyLoss()(predicted_actions, actions)

        self.sl_optimizer.zero_grad()
        loss.backward()
        self.sl_optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())
# Initialize environment
env = rlcard.make('leduc-holdem')

# Get state and action sizes
state_size = env.state_shape[0][0]
action_size = env.num_actions

# Initialize NFSP agent
agent = NFSPAgent(state_size, action_size)

# Training loop
for episode in range(EPISODES):
    state, player_id = env.reset()
    print(state)
    done = False
    while not done:
        # Agent selects action
        action = agent.select_action(state['obs'])
        
        # Environment steps
        next_state, next_player_id = env.step(action)
        
        # Determine if the game is over
        done = env.is_over()
        
        # Calculate reward
        if done:
            payoffs = env.get_payoffs()
            reward = payoffs[player_id]
        else:
            reward = 0
        
        # Store experience
        agent.store_experience(state['obs'], action, reward, next_state['obs'], done)
        
        # Update networks
        agent.update_q_network()
        agent.update_sl_network()
        
        # Move to next state
        state = next_state
        player_id = next_player_id

    # Update target network periodically
    if episode % TARGET_UPDATE == 0:
        agent.update_target_network()

    # Decay exploration rate
    agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)

    # Log progress
    if episode % 100 == 0:
        print(f"Episode {episode}, Epsilon: {agent.epsilon:.2f}")
        
