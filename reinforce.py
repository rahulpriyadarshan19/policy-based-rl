# Script to implement REINFORCE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
device = torch.device("cpu")

plt.rcParams["text.usetex"] = True

class Policy(nn.Module):
    
    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers):
        """
        n_state: number of states
        n_actions: number of actions (output size)
        n_nodes: width of each layer
        n_hidden_layers: number of hidden layers
        """
        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        # Neural network with "n_hidden_layers" hidden layers and "n_nodes" width
        # Input: 2 state paremeters. Output: probabilities of the three actions
        self.policy = nn.Sequential(nn.Linear(n_state, n_nodes), nn.ReLU()) 
        for _ in range(n_hidden_layers):
            self.policy.append(nn.Linear(n_nodes, n_nodes))
            self.policy.append(nn.ReLU())
        self.policy.append(nn.Linear(n_nodes, n_actions))
        self.policy.append(nn.Softmax())

    def forward(self, state):
        """Forward pass
        state: input state vector
            Return:
        policy_state: probabilities of each action for the given state
        """
        policy_state = self.policy(state)

        return policy_state

    def select_action(self, state):
        """Selects action based on the policy.
        state: input state vector
            Return:
        action: index of the selected action
        """
        probability = torch.distributions.Categorical(probs = self.forward(state))
        action = probability.sample()

        return action
    
class Reinforce():
    
    def __init__(self, trace_length, discount, 
                 learning_rate, n_nodes, n_hidden_layers):
        
        self.trace_length = trace_length
        self.discount = discount
        self.learning_rate = learning_rate
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers
        
    def generate_trace(self, state, policy, env):
        """ Generate one random trace in the environment
        state: initial state
        policy: current policy
        env: environment
            Return:
        states: states encountered in the trace
        actions: actions taken in the trace
        rewards: rewards recieved after taking an action in the trace
        """
        states = [state]
        actions = []
        rewards = []

        # done = False

        # Trace ends when the goal has been reached (terminated)
        # or when it has reached a certain length (truncated ) 
        # This length is the min("self.trace_length", max episode length of the environment)
        for _ in range(self.trace_length):
        # while not done:
            action = policy.select_action(state).item()
            state, reward, terminated, truncated, _ = env.step(action)
            state = torch.tensor(state, device = device)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        return states, actions, rewards

        
    def reinforce(self, env, epsilon, n_traces):
        """Function to implement REINFORCE.
        env: environment of the problem
        epsilon: threshold for the convergence
        n_traces: amount of traces considered in one policy optimization
            Return:
        mean_reward_per_batch: mean total reward of all traces in a batch
        """
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n

        policy = Policy(n_states, n_actions, self.n_nodes, self.n_hidden_layers)
        optimizer = torch.optim.Adam(policy.parameters(), lr = self.learning_rate)

        done = False
        mean_reward_per_batch = []
        
        while not done:

            total_reward_per_trace = [] # sum of rewards for every trace in the batch
            loss_per_trace = [] # loss of every trace in the batch, loss = gradient 

            # Randomly sample "n_traces" traces and calculate their collective loss
            for _ in range(n_traces):

                state, _ = env.reset()
                state = torch.tensor(state, device = device)

                states, actions, rewards = self.generate_trace(state, policy, env)

                total_reward_per_trace.append(np.sum(rewards))
                
                T = len(states)
                value_estimate = 0
                value_estimate_per_state = np.empty(T - 1)

                for k in np.arange(T - 2, -1, -1):

                    value_estimate = rewards[k] + self.discount * value_estimate
                    value_estimate_per_state[k] = value_estimate

                # Convert NumPy arrays to torch Tensors to allow for vectorized operations
                states = torch.stack(states)
                actions = torch.LongTensor(actions)
                value_estimate_per_state = torch.Tensor(value_estimate_per_state)

                policies =  policy(states[:-1]) # pi(s)
                selected_policies = policies.gather(dim = 1, index = actions.reshape(actions.size(dim = 0), 1)).squeeze() # pi(a|s)
                log_policies = torch.log(selected_policies) # log(pi(a|s))

                loss_per_trace.append(torch.sum(value_estimate_per_state * log_policies + self.learning_rate*(torch.sum(-selected_policies*log_policies))))

            mean_reward_per_batch.append(np.mean(total_reward_per_trace))

            # Convert NumPy array to torch Tensors to update the policy's parameters
            loss_per_trace = torch.stack(loss_per_trace)
            mean_loss_per_batch = -torch.mean(loss_per_trace)

            optimizer.zero_grad()
            mean_loss_per_batch.backward()
            optimizer.step()

            print(f"Mean reward: {mean_reward_per_batch[-1]}, Loss: {torch.abs(mean_loss_per_batch)}")

            # Check for convergence
            if torch.abs(mean_loss_per_batch) < epsilon:
                done = True

        return mean_reward_per_batch    

if __name__ in "__main__":
    
    mean_rewards_per_batch = [] # Averaging mean reward over many runs
    lengths = []    # Length of each run
    
    # Averaging over 20 runs
    for i in tqdm(range(20)):
    
        # Making the environment
        env = gym.make("Acrobot-v1")
        reinforce_init = Reinforce(trace_length = 500, discount = 0.99, 
                                learning_rate = 0.05, n_nodes = 32, n_hidden_layers = 1) 
        mean_reward_per_batch = np.array(reinforce_init.reinforce(env, epsilon = 0.1, n_traces = 12))
        mean_rewards_per_batch.append(mean_reward_per_batch)
        lengths.append(len(mean_reward_per_batch))
        env.close()

    # Taking the shortest length of all episodes because each episode is of different length
    shortest_length = int(np.min(np.array(lengths, dtype = np.float32)))
    mean_rewards_per_batch = np.array([mean_reward[:shortest_length] for mean_reward in mean_rewards_per_batch], dtype = np.float32)
    mean_rewards_per_batch = np.mean(mean_rewards_per_batch, axis = 0)
    plt.plot(np.arange(shortest_length), mean_rewards_per_batch)
    plt.xlabel("Shortest length of episode")
    plt.ylabel("Mean reward per episode averaged over 20 runs")
    plt.show()
        
 

