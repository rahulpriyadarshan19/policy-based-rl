# Script to implement REINFORCE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import gymnasium as gym
device = torch.device("cpu")

class Policy(nn.Module):
    
    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers):
        """n_state: number of states
        n_actions: number of actions (output size)
        n_nodes: width of each layer
        n_hidden_layers: number of hidden layers
        """
        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 
        self.n_actions = n_actions

        # Neural network with "n_hidden_layers" hidden layers
        # Input: 4 state paremeters. Output: Q-values for the two actions
        self.policy = nn.Sequential(nn.Linear(n_state, n_nodes), nn.ReLU()) 
        for _ in range(n_hidden_layers):
            self.policy.append(nn.Linear(n_nodes, n_nodes))
            self.policy.append(nn.ReLU())
        self.policy.append(nn.Linear(n_nodes, n_actions))
        self.policy.append(nn.Softmax())

    def forward(self, state):
        """Forward pass
        state: input state vector
        """
        Q_state = self.policy(state)
        return Q_state

    def select_action(self, state):
        """Selects action based on the policy.
        """
        prob = torch.distributions.Categorical(probs = self.forward(state))
        a = prob.sample()
        return a
    
class Reinforce():
    
    def __init__(self, trace_length, discount, learning_rate, 
                 n_nodes, n_hidden_layers, n_states, n_actions):
        self.trace_length = trace_length
        self.discount = discount
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers
        self.n_states = n_states
        self.n_actions = n_actions
        
    def generate_trace(self, state, policy, env):
        """state: initial state
        policy: current policy
        env: environment
        """
        states = [state]
        actions = []
        rewards = []
        for i in range(self.trace_length):
            action = policy.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(state, device = device)
            actions.append(action)
            states.append(state)
            rewards.append(reward)
            if terminated or truncated:
                break
            
        rewards = np.array(rewards, dtype = np.float32)
        return states, actions, rewards

        
    def reinforce(self, epsilon, n_episodes):
        """Function to implement REINFORCE.
        """
        env = gym.make("MountainCar-v0")
        state, _ = env.reset()
        state = torch.tensor(state, device = device)
        policy = Policy(self.n_states, self.n_actions, self.n_nodes, self.n_hidden_layers)
        
        for episode in range(n_episodes):
            states, actions, rewards = self.generate_trace(state, policy, env)
            
            
        
        
        
    

    

if __name__ in "__main__":
    
    # Making the environment
    env = gym.make("MountainCar-v0")
    state = torch.tensor(np.array([0.1, 0.9], dtype = np.float32), device = device) 
    policy = Policy(n_state = 2, n_actions = 3, n_nodes = 32, n_hidden_layers = 4)
    print(policy.select_action(state))
    
    
    
