import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'done'))

class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.transition = Transition

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self, batch_size):
        if len(self.memory) < batch_size:
            return None
        transitions = random.sample(self.memory, batch_size)
        batch = self.transition(*zip(*transitions))
        
        states = torch.cat(batch.state)
        next_states = torch.cat(batch.next_state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        dones = torch.cat(batch.done)
        
        return states, next_states, actions, rewards, dones

class Dqn:
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.target_net = Network(input_size, nb_action)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.steps = 0
        self.update_target = 500

    def select_action(self, state):
        self.steps += 1
        epsilon = max(0.05, 0.5 * (0.995 ** (self.steps // 200)))
        
        if random.random() < epsilon:
            return random.randint(0, self.model.nb_action - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state)
                action_index = q_values.max(1)[1].item()
                return max(0, min(action_index, self.model.nb_action - 1))

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action, batch_done):
        valid_mask = (batch_action >= 0) & (batch_action < self.model.nb_action)
        
        if not valid_mask.all():
            batch_state = batch_state[valid_mask]
            batch_next_state = batch_next_state[valid_mask]
            batch_reward = batch_reward[valid_mask]
            batch_action = batch_action[valid_mask]
            batch_done = batch_done[valid_mask]
            
            if len(batch_action) == 0:
                return
        
        batch_action = batch_action.long()
        current_q = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_q = self.target_net(batch_next_state).detach().max(1)[0]
        next_q = next_q * (1 - batch_done.float())  # Zero out next_q if done
        target_q = batch_reward + self.gamma * next_q
        loss = F.smooth_l1_loss(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        if self.steps % self.update_target == 0:
            self.target_net.load_state_dict(self.model.state_dict())

    def update(self, reward, new_signal, done):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        action = self.select_action(new_state)
        action = max(0, min(action, self.model.nb_action - 1))
        
        self.memory.push(
            self.last_state, 
            new_state, 
            torch.LongTensor([action]), 
            torch.Tensor([self.last_reward]),
            torch.BoolTensor([done])
        )
        
        if len(self.memory.memory) > 1000:
            batch = self.memory.sample(128)
            if batch is not None:
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = batch
                self.learn(batch_state, batch_next_state, batch_reward, batch_action, batch_done)
        
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def save(self):
        if not os.path.exists('models'):
            os.makedirs('models')
        torch.save({
            'state_dict': self.model.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, 'models/last_brain.pth')
        print("Model saved to models/last_brain.pth")

    def load(self):
        if os.path.isfile('models/last_brain.pth'):
            print('Loading model...')
            checkpoint = torch.load('models/last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps = checkpoint.get('steps', 0)
            print('Model loaded successfully!')
        else:
            print('No model found! Creating new model.')