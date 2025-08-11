# ---------------agent.py---------------------
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import time

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

Transition = namedtuple(
    "Transition", ("state", "next_state", "action", "reward", "done")
)


class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        # Enhanced network architecture for better GPU utilization
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, nb_action)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
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

        # Move all tensors to GPU
        states = torch.cat(batch.state).to(device)
        next_states = torch.cat(batch.next_state).to(device)
        actions = torch.cat(batch.action).to(device)
        rewards = torch.cat(batch.reward).to(device)
        dones = torch.cat(batch.done).to(device)

        return states, next_states, actions, rewards, dones

    def __len__(self):
        return len(self.memory)


class DqnGPU:
    def __init__(
        self, input_size, nb_action, gamma=0.99, lr=0.0001, memory_capacity=100000
    ):
        self.gamma = gamma
        self.input_size = input_size
        self.nb_action = nb_action
        self.reward_window = []

        # Initialize networks on GPU
        self.model = Network(input_size, nb_action).to(device)
        self.target_net = Network(input_size, nb_action).to(device)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

        # Memory and optimizer
        self.memory = ReplayMemory(memory_capacity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10000, gamma=0.95
        )

        # Initialize state on GPU
        self.last_state = torch.zeros(1, input_size, device=device)
        self.last_action = 0
        self.last_reward = 0
        self.steps = 0
        self.update_target = 1000  # Update target network every 1000 steps
        self.training_start = 1000  # Start training after 1000 experiences

        # Performance tracking
        self.episode_rewards = []
        self.losses = []

    def select_action(self, state, training=True):
        self.steps += 1

        if training:
            # Epsilon-greedy with decay
            epsilon = max(0.01, 0.5 * (0.995 ** (self.steps // 500)))

            if random.random() < epsilon:
                return random.randint(0, self.nb_action - 1)

        # Greedy action selection
        with torch.no_grad():
            state_tensor = state.to(device)
            q_values = self.model(state_tensor)
            action_index = q_values.max(1)[1].item()
            return max(0, min(action_index, self.nb_action - 1))

    def learn(
        self, batch_state, batch_next_state, batch_reward, batch_action, batch_done
    ):
        # Validate actions
        valid_mask = (batch_action >= 0) & (batch_action < self.nb_action)

        if not valid_mask.all():
            batch_state = batch_state[valid_mask]
            batch_next_state = batch_next_state[valid_mask]
            batch_reward = batch_reward[valid_mask]
            batch_action = batch_action[valid_mask]
            batch_done = batch_done[valid_mask]

            if len(batch_action) == 0:
                return 0

        batch_action = batch_action.long()

        # Current Q values
        current_q = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )

        # Next Q values from target network
        with torch.no_grad():
            next_q = self.target_net(batch_next_state).max(1)[0]
            next_q = next_q * (1 - batch_done.float())  # Zero out next_q if done
            target_q = batch_reward + self.gamma * next_q

        # Compute loss
        loss = F.huber_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        if self.steps % self.update_target == 0:
            self.target_net.load_state_dict(self.model.state_dict())
            print(f"Target network updated at step {self.steps}")

        return loss.item()

    def update(self, reward, new_signal, done):
        # Convert to tensor and move to GPU
        new_state = torch.tensor(
            new_signal, dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Select action
        action = self.select_action(new_state, training=True)
        action = max(0, min(action, self.nb_action - 1))

        # Store transition in memory
        self.memory.push(
            self.last_state.cpu(),
            new_state.cpu(),
            torch.tensor([action], dtype=torch.long),
            torch.tensor([self.last_reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
        )

        # Train if enough experiences
        loss = 0
        if len(self.memory) > self.training_start:
            batch = self.memory.sample(min(128, len(self.memory) // 4))
            if batch is not None:
                (
                    batch_state,
                    batch_next_state,
                    batch_action,
                    batch_reward,
                    batch_done,
                ) = batch
                loss = self.learn(
                    batch_state,
                    batch_next_state,
                    batch_reward,
                    batch_action,
                    batch_done,
                )
                self.losses.append(loss)

        # Update state
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        # Keep reward window manageable
        if len(self.reward_window) > 1000:
            del self.reward_window[0]

        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1)

    def get_epsilon(self):
        return max(0.01, 0.5 * (0.995 ** (self.steps // 500)))

    def save(self, filename="last_brain_gpu.pth"):
        if not os.path.exists("models"):
            os.makedirs("models")

        filepath = os.path.join("models", filename)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "steps": self.steps,
                "reward_window": self.reward_window,
                "episode_rewards": self.episode_rewards,
                "losses": self.losses,
                "input_size": self.input_size,
                "nb_action": self.nb_action,
                "gamma": self.gamma,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load(self, filename="last_brain_gpu.pth"):
        filepath = os.path.join("models", filename)
        if os.path.isfile(filepath):
            print("Loading model...")
            checkpoint = torch.load(filepath, map_location=device)

            self.model.load_state_dict(checkpoint["state_dict"])
            self.target_net.load_state_dict(checkpoint["target_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

            self.steps = checkpoint.get("steps", 0)
            self.reward_window = checkpoint.get("reward_window", [])
            self.episode_rewards = checkpoint.get("episode_rewards", [])
            self.losses = checkpoint.get("losses", [])

            print(f"Model loaded successfully from {filepath}!")
            print(f"Resumed at step: {self.steps}")
        else:
            print(f"No model found at {filepath}! Creating new model.")

    def get_stats(self):
        """Get training statistics"""
        stats = {
            "steps": self.steps,
            "current_score": self.score(),
            "epsilon": self.get_epsilon(),
            "memory_size": len(self.memory),
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        return stats

    def print_stats(self):
        """Print current training statistics"""
        stats = self.get_stats()
        print(
            f"Steps: {stats['steps']:,} | "
            f"Score: {stats['current_score']:.3f} | "
            f"Epsilon: {stats['epsilon']:.3f} | "
            f"Memory: {stats['memory_size']:,} | "
            f"Loss: {stats['avg_loss']:.4f} | "
            f"LR: {stats['learning_rate']:.6f}"
        )


# Example training loop
episode_number = 10000


def train_agent(env_function, episodes=episode_number, max_steps_per_episode=1000):
    """
    Example training function

    Args:
        env_function: Function that returns (state_size, action_size, step_function)
        episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
    """
    # Initialize environment (you'll need to replace this with your environment)
    state_size, action_size = 8, 4  # Example sizes - replace with your environment

    # Initialize agent
    agent = DqnGPU(input_size=state_size, nb_action=action_size)

    # Try to load existing model
    agent.load()

    episode_rewards = []

    for episode in range(episodes):
        # Reset environment (replace with your environment reset)
        state = np.random.randn(state_size)  # Example initial state
        episode_reward = 0
        done = False
        steps = 0

        while not done and steps < max_steps_per_episode:
            # Take action
            action = agent.update(agent.last_reward, state, done)

            # Step environment (replace with your environment step)
            # next_state, reward, done, info = env.step(action)
            next_state = np.random.randn(state_size)  # Example next state
            reward = random.uniform(-1, 1)  # Example reward
            done = random.random() < 0.01  # Example done condition

            episode_reward += reward
            state = next_state
            steps += 1

        episode_rewards.append(episode_reward)

        # Print statistics every 100 episodes
        if episode % 100 == 0:
            agent.print_stats()
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, "
                f"Avg Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}"
            )

        # Save model every 500 episodes
        if episode % 500 == 0:
            agent.save(f"checkpoint_episode_{episode}.pth")

    # Final save
    agent.save("final_model.pth")
    print("Training completed!")
    return agent, episode_rewards


if __name__ == "__main__":
    # Example usage
    print("GPU DQN Agent initialized")

    # You can create and train your agent like this:
    # agent, rewards = train_agent(your_env_function, episodes=2000)

    # Or create an agent directly:
    # agent = DqnGPU(input_size=your_state_size, nb_action=your_action_size)
