# Script to implement REINFORCE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
device = torch.device("cpu")

plt.rcParams["text.usetex"] = True

class NeuralNetwork(nn.Module):
    """
    General neural network. Used for both the policy and the value function.
    """
    
    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers, last_layer):
        """
        n_state: number of states
        n_actions: number of actions (output size)
        n_nodes: width of each layer
        n_hidden_layers: number of hidden layers
        last_layer: activation function in the last layer
        """
        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        # Neural network with "n_hidden_layers" hidden layers and "n_nodes" width
        # Input: 2 state paremeters. Output: probabilities of the three actions
        self.neural_network = nn.Sequential(nn.Linear(n_state, n_nodes), nn.ReLU()) 
        for _ in range(n_hidden_layers):
            self.neural_network.append(nn.Linear(n_nodes, n_nodes))
            self.neural_network.append(nn.ReLU())
        self.neural_network.append(nn.Linear(n_nodes, n_actions))
        self.neural_network.append(last_layer)

    def forward(self, state):
        """Forward pass
        state: input state vector
            Return:
        neural_network_state: output of the neural network
        """
        neural_network_state = self.neural_network(state)

        return neural_network_state

class Policy(NeuralNetwork):

    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers):
        super().__init__(n_state, n_actions, n_nodes, n_hidden_layers, last_layer = nn.Softmax())

    def select_action(self, state):
        """Selects action based on the policy.
        state: input state vector
            Return:
        action: index of the selected action
        """
        probability = torch.distributions.Categorical(probs = super().forward(state))
        action = probability.sample()

        return action
    
class Value_function(NeuralNetwork):

    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers):
        super().__init__(n_state, n_actions, n_nodes, n_hidden_layers, last_layer = nn.Linear())

############################################################################################################################

class Policy_based_RL():

    def __init__(self, env, method, trace_length, discount, n_bootstrapping,
                 learning_rate, eta_entropy, n_nodes, n_hidden_layers):
        
        self.env = env
        self.method = method
        
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers

        self.learning_rate = learning_rate
        self.policy = Policy(self.n_states, self.n_actions, self.n_nodes, self.n_hidden_layers)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)

        # Will be defined later if used
        self.value_function = None 
        self.optimizer_value_function = None

        self.eta_entropy = eta_entropy
        self.n_bootstrapping = n_bootstrapping
        self.trace_length = trace_length
        self.discount = discount
        

    def generate_trace(self, state):
        """ Generate one random trace in the environment
        state: initial state
            Return:
        states: states encountered in the trace
        actions: actions taken in the trace
        rewards: rewards recieved after taking an action in the trace
        """
        states = [state]
        actions = []
        rewards = []

        # Trace ends when the goal has been reached (terminated)
        # or when it has reached a certain length (truncated ) 
        # This length is the min("self.trace_length", max episode length of the environment)
        for _ in range(self.trace_length):

            action = self.policy.select_action(state).item()
            state, reward, terminated, truncated, _ = self.env.step(action)
            state = torch.tensor(state, device = device)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        return states, actions, rewards
    
    def calculate_value_estimate(self, states, rewards):

        T = len(states)
        value_estimate = 0
        value_estimate_per_state = np.empty(T - 1)

        if self.method == "reinforce" or "baseline_subtraction":
            for k in np.arange(T - 2, -1, -1):

                value_estimate = rewards[k] + self.discount * value_estimate
                value_estimate_per_state[k] = value_estimate

            if self.method == "baseline_subtraction":
                value_estimate_per_state - self.value_function(states[:-1])

        elif self.method == "bootstrapping" or self.method == "bootstrapping_basline_subtraction":
            for k in range(T - 1):

                discount_n = np.full(self.n_bootstrapping, self.discount) ** np.arange(self.n_bootstrapping)
                value_estimate = np.sum(discount_n * rewards[k : k + self.n_bootstrapping]) \
                                + (self.discount ** self.n_bootstrapping) * self.value_function(states[k + self.n_bootstrapping])
                value_estimate_per_state[k] = value_estimate

            if self.method == "bootstrapping_basline_subtraction":
                value_estimate_per_state - self.value_function(states[:-1])

        return value_estimate_per_state
    
    def general_policy_based_algorithm(self, epsilon, n_traces):
        """
        epsilon: threshold for the convergence
        n_traces: amount of traces considered in one policy optimization
        """

        done = False
        mean_reward_per_batch = []
        
        while not done:

            total_reward_per_trace = [] # sum of rewards for every trace in the batch
            policy_loss_per_trace = [] # loss of every trace in the batch, loss = gradient 
            value_function_loss_per_trace = [] # loss of every trace in the batch, loss = gradient 

            # Randomly sample "n_traces" traces and calculate their collective loss
            for _ in range(n_traces):

                state, _ = self.env.reset()
                state = torch.tensor(state, device = device)

                states, actions, rewards = self.generate_trace(state)

                total_reward_per_trace.append(np.sum(rewards))

                # Convert NumPy arrays to torch Tensors to allow for vectorized operations
                states = torch.stack(states)
                
                value_estimate_per_state = self.calculate_value_estimate(states, rewards)

                # Convert NumPy arrays to torch Tensors to allow for vectorized operations
                actions = torch.LongTensor(actions)
                value_estimate_per_state = torch.Tensor(value_estimate_per_state)

                # Calculate loss of policy network
                policies = self.policy(states[:-1]) # pi(s)
                selected_policies = policies.gather(dim = 1, index = actions.reshape(actions.size(dim = 0), 1)).squeeze() # pi(a|s)
                log_policies = torch.log(selected_policies) # log(pi(a|s))

                policy_loss_per_trace.append(torch.sum(value_estimate_per_state * log_policies) + self.eta_entropy*(torch.sum(-selected_policies*log_policies)))

                if self.method != "reinforce":
                    # Calculate loss of value function network

                    if self.method == "bootstrapping":
                        value_function_loss_per_trace.append(torch.nn.MSELoss(value_estimate_per_state, self.value_function(states[:-1])) + self.eta_entropy*(torch.sum(-selected_policies*log_policies)))
                    
                    else:
                        value_function_loss_per_trace.append(torch.mean(torch.square(value_estimate_per_state)) + self.eta_entropy*(torch.sum(-selected_policies*log_policies)))

            mean_reward_per_batch.append(np.mean(total_reward_per_trace))

            # Convert NumPy array to torch Tensors to update the policy's parameters
            policy_loss_per_trace = torch.stack(policy_loss_per_trace)
            mean_policy_loss_per_batch = -torch.mean(policy_loss_per_trace)

            self.optimizer_policy.zero_grad()
            mean_policy_loss_per_batch.backward()
            self.optimizer_policy.step()

            if self.method != "reinforce":
                value_function_loss_per_trace = torch.stack(value_function_loss_per_trace)
                mean_value_function_loss_per_batch = -torch.mean(value_function_loss_per_trace)

                self.optimizer_value_function.zero_grad()
                mean_value_function_loss_per_batch.backward()
                self.optimizer_value_function.step()

            # print(f"Mean reward: {mean_reward_per_batch[-1]}, Loss: {torch.abs(mean_policy_loss_per_batch)}")

            # Check for convergence
            if  self.method == "reinforce" and torch.abs(mean_policy_loss_per_batch) < epsilon:
                done = True

            elif self.method != "reinforce" and (torch.abs(mean_policy_loss_per_batch) and torch.abs(mean_value_function_loss_per_batch)) < epsilon:
                done = True

        return mean_reward_per_batch   

if __name__ in "__main__":
    
    mean_rewards_per_batch = [] # Averaging mean reward over many runs
    lengths = []    # Length of each run
    
    # Averaging over 20 runs
    for i in tqdm(range(20)):
    
        # Making the environment
        env = gym.make("Acrobot-v1")
        policy_based_init = Policy_based_RL(env, method = "reinforce", trace_length = 500,
                                         discount = 0.99, n_bootstrapping = 3,
                                         learning_rate = 0.05, eta_entropy = 0.01, n_nodes = 32, n_hidden_layers = 1)
        mean_reward_per_batch = np.array(policy_based_init.general_policy_based_algorithm(epsilon = 0.1, n_traces = 12))
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
        
 

