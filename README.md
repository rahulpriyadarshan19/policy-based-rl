This is Group 24's repository for assignment 3 of the Reinforcement Learning course.

In this assignment, we implement REINFORCE and Actor-Critic algorithms with baseline subtraction, bootstrapping and both on the Acrobot environment. 

In order to recreate our hyperparameter tuning experiment for REINFORCE, use the command "python policy_based_rl.py -- hyperparameter_tuning".

In order to train all algorithms (REINFORCE and the three Actor-Critic algorithms) using the best-performing model parameters of REINFORCE,  use the command "python policy_based_rl.py --optimal_performance".

In order to determine the policy gradient's variance for all algorithms, use the command "python policy_based_rl.py --policy_gradient_variance".
