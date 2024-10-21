import torch
import torch.nn as nn
import torch.optim as optim
import random
import json
import tqdm

# Define DQN Network
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        if state_dim == 0:
            raise ValueError("State dimension must be greater than zero.")
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, state):
        return self.fc(state)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Training Function
def train_dqn(dqn, memory, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])

    q_values = dqn(state_batch).gather(1, action_batch)
    next_q_values = dqn(next_state_batch).max(1)[0].detach()
    expected_q_values = reward_batch + (gamma * next_q_values)

    loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Function to run DQN Optimization
def run_dqn_optimization(feedback, design_params):
    # Ensure design_params is a dictionary
    if isinstance(design_params, str):
        try:
            design_params = json.loads(design_params)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for design parameters.")

    state_values = [float(value) for key, value in design_params.items() if isinstance(value, (int, float))]
    if len(state_values) == 0:
        # Provide default state values if none are present
        state_values = [1.0]  # Default placeholder value
    state_dim = len(state_values)  # Example number of state features
    action_dim = 3  # Example number of actions (e.g., reduce weight, increase strength)
    batch_size = 32
    gamma = 0.99
    learning_rate = 0.001
    num_episodes = 100

    # Initialize DQN, optimizer, and replay memory
    dqn = DQN(state_dim, action_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
    memory = ReplayMemory(10000)

    # Example initial state (placeholder, should be based on design_params)
    initial_state = torch.tensor(state_values, dtype=torch.float32)

    # Training Loop for DQN optimization
    for episode in tqdm.tqdm(range(num_episodes)):
        state = initial_state.unsqueeze(0)  # Add batch dimension
        for t in range(100):  # Limit each episode to 100 steps
            # Select action (epsilon-greedy)
            epsilon = 0.1
            if random.random() > epsilon:
                with torch.no_grad():
                    action = dqn(state).argmax().view(1, 1)
            else:
                action = torch.tensor([[random.randrange(action_dim)]], dtype=torch.long)

            # Execute action and observe new state and reward (placeholder logic)
            next_state = state  # Update with actual next state logic
            reward = torch.tensor([1.0], dtype=torch.float32)  # Update with actual reward based on feedback

            # Store transition in memory
            memory.push((state, action, reward, next_state))

            # Move to next state
            state = next_state

            # Train DQN
            train_dqn(dqn, memory, optimizer, batch_size, gamma)

    # Placeholder for optimized design parameters
    optimized_design = {
        "Type": design_params.get("Type", "Unknown"),
        "Load Capacity": design_params.get("Load Capacity", "Unknown"),
        "Features": design_params.get("Features", "Unknown"),
        "Optimized": "Yes"
    }

    return optimized_design
