# Script to implement REINFORCE
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
from scipy.signal import savgol_filter
device = torch.device("cpu")

# Pretty plots and a color-blind friendly color scheme
plt.rcParams["text.usetex"] = True
colors = {"blue":"#4477aa", "green":"#228833", "red":"#ee6677"}

class NeuralNetwork(nn.Module):
    """
    General neural network. Used for both the policy and the value function.
    """
    
    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers, last_layer = None):
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
        if last_layer is not None:
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
        super().__init__(n_state, n_actions, n_nodes, n_hidden_layers, last_layer = nn.Softmax(dim = -1))

    def select_action(self, state, exploration = "on"):
        """Selects action based on the policy.
        state: input state vector
        exploration: indicates whether exploration is on or off
            Return:
        action: index of the selected action
        """
        if exploration == "on":
            probability = torch.distributions.Categorical(probs = super().forward(state))
            action = probability.sample()
        elif exploration == "off":
            probability = super().forward(state).argmax()
            action = probability.max()

        return action
    
    def evaluate(self, eval_env, n_eval_episodes = 30, max_episode_length = 100):
        """Evaluate the policy using an evaluation environment
        eval_env: evaluation environment
        n_eval_episodes: number of evaluation intervals
        max_episode_length: maximum amount of environment steps within one episode 
        """
        total_rewards = []  # list to store the reward per episode
        for idx in range(n_eval_episodes):

            state, _ = eval_env.reset()
            state = torch.tensor(state, device = device)
            total_reward = 0

            for _ in range(max_episode_length):

                action = self.select_action(state, exploration = "off").item()
                state, reward, terminated, truncated, _ = eval_env.step(action)
                state = torch.tensor(state, device = device)
                total_reward += reward

                if terminated or truncated:
                    break
                
            total_rewards.append(total_reward)
        mean_reward = np.mean(total_rewards)
        return mean_reward
    
class Value_function(NeuralNetwork):

    def __init__(self, n_state, n_actions, n_nodes, n_hidden_layers):
        super().__init__(n_state, n_actions, n_nodes, n_hidden_layers)

############################################################################################################################

class Policy_based_RL():

    def __init__(self, env, eval_env, method, trace_length, 
                 discount, n_bootstrapping, learning_rate, 
                 eta_entropy, n_nodes, n_hidden_layers):
        
        self.env = env
        self.eval_env = eval_env
        self.method = method
        
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_nodes = n_nodes
        self.n_hidden_layers = n_hidden_layers

        self.learning_rate = learning_rate
        self.policy = Policy(self.n_states, self.n_actions, self.n_nodes, self.n_hidden_layers)
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr = self.learning_rate)

        if method == "reinforce":
            self.value_function = None 
            self.optimizer_value_function = None
        else:
            self.value_function = Value_function(self.n_states, 1, self.n_nodes, self.n_hidden_layers)
            self.optimizer_value_function = torch.optim.Adam(self.value_function.parameters(), lr = self.learning_rate)

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
                value_estimate_per_state = value_estimate_per_state - self.value_function(states[:-1]).squeeze()

        elif self.method == "bootstrapping" or self.method == "bootstrapping_basline_subtraction":
            for k in range(T - 1):

                discount_n = np.full(self.n_bootstrapping, self.discount) ** np.arange(self.n_bootstrapping)
                value_estimate = np.sum(discount_n * rewards[k : k + self.n_bootstrapping]) \
                                + (self.discount ** self.n_bootstrapping) * self.value_function(states[k + self.n_bootstrapping]).squeeze()
                value_estimate_per_state[k] = value_estimate

            if self.method == "bootstrapping_basline_subtraction":
                value_estimate_per_state = value_estimate_per_state - self.value_function(states[:-1]).squeeze()

        return value_estimate_per_state
    
    def general_policy_based_algorithm(self, n_timesteps, n_traces, eval_interval):
        """Train a policy using REINFORCE or Actor-Critic
        n_timesteps: total amount of environment steps taken during training
        n_traces: amount of traces considered in one policy optimization
        eval_interval: amount of environment steps after which the policy is evaluated
            Return:
        eval_timesteps: environment steps when the policy was evaluated
        eval_mean_reward: average rewards of the policy on the evaluation environment
        """
        
        eval_timesteps = []
        eval_mean_reward = []

        for time in range(int(n_timesteps / (n_traces  * self.trace_length))):

            policy_loss_per_trace = [] # loss of every trace in the batch, loss = gradient 
            value_function_loss_per_trace = [] # loss of every trace in the batch, loss = gradient 

            # Randomly sample "n_traces" traces and calculate their collective loss
            for _ in range(n_traces):

                state, _ = self.env.reset()
                state = torch.tensor(state, device = device)

                states, actions, rewards = self.generate_trace(state)

                # Convert NumPy arrays to torch Tensors to allow for vectorized operations
                states = torch.stack(states)
                # states = states.clone().detach()
                
                value_estimate_per_state = self.calculate_value_estimate(states, rewards)

                # Convert NumPy arrays to torch Tensors to allow for vectorized operations
                actions = torch.LongTensor(actions)
                value_estimate_per_state = torch.Tensor(value_estimate_per_state)

                # actions = actions.clone().detach()
                # value_estimate_per_state = value_estimate_per_state.clone().detach()

                # Calculate loss of policy network
                policies = self.policy(states[:-1]) # pi(s)
                selected_policies = policies.gather(dim = 1, index = actions.reshape(actions.size(dim = 0), 1)).squeeze() # pi(a|s)
                log_policies = torch.log(selected_policies) # log(pi(a|s))

                policy_loss_per_trace.append(torch.sum(value_estimate_per_state * log_policies) + self.eta_entropy*(torch.sum(-selected_policies*log_policies)))

                if self.method != "reinforce":
                    # Calculate loss of value function network

                    if self.method == "bootstrapping":
                        # value_function_loss_per_trace.append(torch.mean(torch.square(value_estimate_per_state.detach() - self.value_function(states[:-1]).squeeze())))
                        value_function_loss_per_trace.append(torch.mean(torch.square(value_estimate_per_state.detach() - self.value_function(states[:-1]).squeeze().detach())))
                    else:
                        value_function_loss_per_trace.append(torch.mean(torch.square(value_estimate_per_state.detach())))

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

                mean_value_function_loss_per_batch.requires_grad = True
                mean_value_function_loss_per_batch.backward()

                self.optimizer_value_function.step()

            if time % int(eval_interval / (n_traces * self.trace_length)):
                mean_reward = self.policy.evaluate(self.eval_env, n_eval_episodes = 5 * n_traces, max_episode_length = self.trace_length)
                
                eval_timesteps.append(time * n_traces * self.trace_length)
                eval_mean_reward.append(mean_reward)

        print(f"Number of env steps: {time * n_traces * self.trace_length}, Loss: {torch.abs(mean_policy_loss_per_batch)}")

            # Check for convergence
            # if  self.method == "reinforce" and torch.abs(mean_policy_loss_per_batch) < epsilon:
            #     done = True

            # elif self.method != "reinforce" and (torch.abs(mean_policy_loss_per_batch) and torch.abs(mean_value_function_loss_per_batch)) < epsilon:
            #     done = True

        return  eval_timesteps, eval_mean_reward
    
def averaged_runs(hyperparam, hyperparam_array, hyperparam_label, method = "reinforce",
                  best = False):
    """Function to generate results averaging over 20 runs and for hyperparameter optimization.
    hyperparam: Hyperparameter to be optimized (string)
    hyperparam_array: Array of possible hyperparameters
    method: Either reinforce, bootstrapping, baseline_subtraction or bootstrapping_baseline_subtraction
    hyperparam_label: Label of hyperparameter, useful in plotting
    """
    fig, ax = plt.subplots(nrows = 1, ncols = 1)
    color_strings = ["blue", "green", "red"]
    
    if not best:
        for value, color_string in zip(hyperparam_array, color_strings):
            
            mean_eval_timesteps = [] # Averaging mean environment steps over many runs
            mean_eval_rewards = [] # Averaging mean reward over many runs
            lengths = []    # Length of each run
            print(f"{hyperparam} = {value}: ")
            
            for i in tqdm(range(5)):
                
                # Making the environment
                env = gym.make("Acrobot-v1")
                eval_env = gym.make("Acrobot-v1")
            
                # Defining the policy based RL class based on which hyperparameter needs to be tuned
                if hyperparam == "learning_rate":
                    policy_based_init = Policy_based_RL(
                        env, eval_env, method = "reinforce", trace_length = 200,
                        discount = 0.99, n_bootstrapping = 3, learning_rate = value, 
                        eta_entropy = 0.01, n_nodes = 32, n_hidden_layers = 1)
                    
                elif hyperparam == "n_nodes":
                    policy_based_init = Policy_based_RL(
                        env, eval_env, method = "reinforce", trace_length = 200,
                        discount = 0.99, n_bootstrapping = 3, learning_rate = 0.005, 
                        eta_entropy = 0.01, n_nodes = value, n_hidden_layers = 1)
                    
                elif hyperparam == "n_hidden_layers":
                    policy_based_init = Policy_based_RL(
                        env, eval_env, method = "reinforce", trace_length = 200,
                        discount = 0.99, n_bootstrapping = 3, learning_rate = 0.005, 
                        eta_entropy = 0.01, n_nodes = 32, n_hidden_layers = value)
                    
                elif hyperparam == "n_bootstrapping":
                    policy_based_init = Policy_based_RL(
                        env, eval_env, method = "bootstrapping", trace_length = 200,
                        discount = 0.99, n_bootstrapping = value, learning_rate = 0.005, 
                        eta_entropy = 0.03, n_nodes = 32, n_hidden_layers = 1)
                    
                elif hyperparam == "eta_entropy":
                    policy_based_init = Policy_based_RL(
                        env, eval_env, method = "reinforce", trace_length = 200,
                        discount = 0.99, n_bootstrapping = 3, learning_rate = 0.005, 
                        eta_entropy = value, n_nodes = 32, n_hidden_layers = 1)
                    
                # Training the policy-based algorithm and evaluating it on an environment
                eval_timesteps, eval_mean_reward = np.array(policy_based_init.general_policy_based_algorithm
                                                            (n_timesteps = 803000, 
                                                            n_traces = 5, 
                                                            eval_interval = 3000))
                
                mean_eval_timesteps.append(eval_timesteps)
                mean_eval_rewards.append(eval_mean_reward)

                env.close()
                eval_env.close()
                
            # Taking the shortest length of all episodes because each episode is of different length
            mean_eval_timesteps = np.mean(mean_eval_timesteps, axis = 0)
            mean_eval_rewards = np.mean(mean_eval_rewards, axis = 0)
            
            # Smoothing the rewards
            mean_eval_rewards = savgol_filter(mean_eval_rewards, window_length = 5,
                                            polyorder = 2)
            
            # Plotting learning curves
            ax.grid()
            ax.plot(mean_eval_timesteps, mean_eval_rewards, label = f"{hyperparam_label} = {value}",
                    color = colors[color_string])
            
        ax.set_xlabel("Number of environment steps")
        ax.set_ylabel("Mean reward of evaluation environment")
        ax.set_title(f"Tuning {hyperparam}")
        ax.legend()
        fig.savefig(f"learning_curve_{method}_{hyperparam}.png", dpi = 300)
        
    elif best:
        mean_eval_timesteps = [] # Averaging mean environment steps over many runs
        mean_eval_rewards = [] # Averaging mean reward over many runs
        
        for i in tqdm(range(20)):
            # Making the environment
            env = gym.make("Acrobot-v1")
            eval_env = gym.make("Acrobot-v1")
            
            policy_based_init = Policy_based_RL(
                env, eval_env, method = method, trace_length = 200,
                discount = 0.99, n_bootstrapping = 3, learning_rate = 0.005, 
                eta_entropy = 0.03, n_nodes = 32, n_hidden_layers = 1)
            
            # Training the policy-based algorithm and evaluating it on an environment
            eval_timesteps, eval_mean_reward = np.array(policy_based_init.general_policy_based_algorithm
                                                        (n_timesteps = 803000, 
                                                        n_traces = 5, 
                                                        eval_interval = 3000))
            
            mean_eval_timesteps.append(eval_timesteps)
            mean_eval_rewards.append(eval_mean_reward)

            env.close()
            eval_env.close()
            
        # Taking the shortest length of all episodes because each episode is of different length
        mean_eval_timesteps = np.mean(mean_eval_timesteps, axis = 0)
        mean_eval_rewards = np.mean(mean_eval_rewards, axis = 0)
        
        # Smoothing the rewards
        mean_eval_rewards = savgol_filter(mean_eval_rewards, window_length = 5,
                                        polyorder = 2)
        
        ax.grid()
        ax.plot(mean_eval_timesteps, mean_eval_rewards, color = colors["blue"])
        ax.set_xlabel("Number of environment steps")
        ax.set_ylabel("Mean reward of evaluation environment")
        ax.set_title(f"Performance of {method}")
        fig.savefig(f"performance_{method}.png", dpi = 300)
            
    
if __name__ in "__main__":
    
    hyperparam_dict = {
        "learning_rate": [np.array([5e-3, 1e-2, 5e-2], dtype = np.float32), r"$\alpha$"],
        "n_nodes": [np.array([32, 64, 128], dtype = int), "Nodes"],
        "n_hidden_layers": [np.array([1, 2, 3], dtype = int), "HiddenLayers"],
        "n_bootstrapping": [np.array([1, 3, 5], dtype = int), "n-step"],
        "eta_entropy": [np.array([0.003, 0.01, 0.03], dtype = np.float32), r"$\eta$"]
    }
    # print(list(hyperparam_dict.keys()))
    # for hyperparam in list(hyperparam_dict.keys())[3:4]:
    #     averaged_runs(hyperparam, 
    #                   hyperparam_array = hyperparam_dict[hyperparam][0], 
    #                   hyperparam_label = hyperparam_dict[hyperparam][1], 
    #                   method = "reinforce")
        
    # Getting the best runs of bootstrapping and bootstrapping_baseline_subtraction
    hyperparam = list(hyperparam_dict.keys())[0]
    averaged_runs(hyperparam, hyperparam_dict[hyperparam][0], hyperparam_dict[hyperparam][1],
                  method = "bootstrapping", best = True)
    averaged_runs(hyperparam, hyperparam_dict[hyperparam][0], hyperparam_dict[hyperparam][1],
                  method = "bootstrapping_baseline_subtraction", best = True)
    
    
        
 