import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR  # Import the scheduler
import numpy as np
import random
from collections import deque
import os
import matplotlib.pyplot as plt
from matplotlib import cm  # For color maps
from line_env import SupervisorLineEnv
import copy
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Define the neural network for value function approximation
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_layers, activation_functions):
        super(ValueNetwork, self).__init__()
        layers = []
        input_dim = state_dim

        # Build hidden layers
        for idx, (hidden_dim, activation) in enumerate(zip(hidden_layers, activation_functions)):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'GELU':
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation}")
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        # Flatten the buffer to get all experiences
        all_experiences = [experience for trajectory in self.buffer for experience in trajectory]
        return random.sample(all_experiences, min(len(all_experiences), batch_size))

    def __len__(self):
        return len(self.buffer)

# Policy Evaluation using Neural Network Function Approximation
def evaluate_policy(env, save_name, policy_index, value_network, optimizer,
                    scheduler, buffer, gamma=0.99, num_episodes=1000,
                    batch_size=64, return_threshold=0.05,
                    episode_interval=25, get_state_values=None,
                    use_multiple_starts=False):
    mse_loss = nn.MSELoss()
    max_episode_length = env._max_episode_steps

    # Lists to store losses and episodes
    losses = []
    episodes_list = []

    # Prepare to collect value function data over iterations
    value_function_iterations = []
    plot_intervals = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        trajectory = []
        total_reward = 0

        # Collect a trajectory
        while not done:
            action = policy_index
            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            trajectory.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done
            })

            state = next_state

            if len(trajectory) >= max_episode_length:
                break

        # Store the trajectory in the replay buffer
        buffer.push(trajectory)

        # Compute returns from multiple starting points with the gamma^remaining_steps >= return_threshold condition
        states = []
        targets = []
        for traj in buffer.buffer:
            rewards = [step['reward'] for step in traj]
            trajectory_length = len(rewards)
            if use_multiple_starts:
                for start in range(trajectory_length):
                    remaining_steps = trajectory_length - start
                    discount_factor = gamma ** remaining_steps
                    if (discount_factor <= return_threshold) or \
                            (remaining_steps * 10 / 9 >= trajectory_length):  # use at least 10% of the trajectory
                        G = compute_return(rewards[start:], gamma)
                        state = torch.FloatTensor(traj[start]['state'])
                        states.append(state)
                        targets.append(G)
                    else:
                        # Skip starting points where gamma^remaining_steps > return_threshold
                        break  # Since discount_factor decreases with increasing start, we can break early
            else:
                # Only use the first starting point
                G = compute_return(rewards, gamma)
                state = torch.FloatTensor(traj[0]['state'])
                states.append(state)
                targets.append(G)
        # Update value network using mini-batch gradient descent
        if len(states) >= batch_size:
            optimizer.zero_grad()
            batch_indices = np.random.choice(len(states), batch_size, replace=False)
            batch_states = torch.stack([states[i] for i in batch_indices])
            batch_targets = torch.FloatTensor([targets[i] for i in batch_indices]).unsqueeze(1)
            predictions = value_network(batch_states)
            loss = mse_loss(predictions, batch_targets)
            loss.backward()
            optimizer.step()

            # Update the learning rate scheduler
            scheduler.step()

            # Record the loss
            losses.append(loss.item())
            episodes_list.append(episode + 1)

            # Collect value function data at specified intervals
            if (episode + 1) % episode_interval == 0:
                print(f"Episode {episode+1}/{num_episodes}, Loss: {loss.item():.4f}, Total Reward: {total_reward}")
                print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

                # Evaluate and store the value function over a range of states
                state_values, x_values = get_state_values(env)
                value_function_iterations.append(state_values)
                plot_intervals.append(episode + 1)

    # Save the trained value network
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    torch.save(value_network.state_dict(), save_name)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot the original loss
    ax[0].plot(episodes_list, losses)
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss over Episodes')
    ax[0].grid()

    # Compute the logarithm of the losses
    epsilon = 1e-8  # Small value to prevent log(0)
    log_losses = np.log(np.array(losses) + epsilon)

    # Plot the logarithm of the loss
    ax[1].plot(episodes_list, log_losses)
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Log Loss')
    ax[1].set_title('Log Loss over Episodes')
    ax[1].grid()

    plt.tight_layout()
    plt.savefig('plots/loss_and_log_loss_over_episodes.png')
    plt.show()
    # Plot all value functions on the same figure
    plot_all_value_functions(x_values, value_function_iterations, plot_intervals)

def compute_return(rewards, gamma):
    """
    Compute the total discounted return for a given list of rewards.
    """
    G = 0
    for idx, reward in enumerate(rewards):
        G += (gamma ** idx) * reward
    return G

def policy(q):
    # Policies are embedded in the environment
    return int(q)

def plot_all_value_functions(x_values, value_function_iterations, plot_intervals, title='Value Function Over Iterations', fignumber=1):
    """
    Plot all value function curves on the same figure.

    Args:
        x_values (array-like): The range of state values.
        value_function_iterations (list): List of value functions at different iterations.
        plot_intervals (list): Episodes at which the value functions were recorded.
    """
    plt.figure(fignumber, figsize=(10, 6))
    num_plots = len(value_function_iterations)
    colors = cm.viridis(np.linspace(0, 1, num_plots))

    for idx, state_values in enumerate(value_function_iterations):
        plt.plot(x_values, state_values, color=colors[idx],
                 label=f'Episode {plot_intervals[idx]}')

    plt.xlabel('State Dimension 0')
    plt.ylabel('Value')
    plt.title(title)
    plt.grid()
    # plt.savefig('plots/value_function_over_iterations.png')
    # plt.show()

def get_state_values_lineenv(env):
    state_values = []
    x_values = np.linspace(env.MINIMAL_VALUE_POSITION_X, env.MAXIMAL_VALUE_POSITION_X, 200)
    for x in x_values:
        # Adjust according to your environment's state representation
        state = env.get_observations(x)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            value = value_network(state_tensor).item()
        state_values.append(value)
    return state_values, x_values


def get_true_values(Env, env_kwargs, resolution=300):
    # plot true value fun
    INITIAL_STATES = np.linspace(-3, 3, resolution)
    Vqs = np.zeros((2, resolution), dtype=np.float32)
    for q in [0,1]:#, 2, 3]:
        Vq = []
        for index, INITIAL_STATE in enumerate(INITIAL_STATES):
            discounted_rewards = 0
            steps = 0
            env = Env(INITIAL_STATE=INITIAL_STATE, **env_kwargs)
            env.reset()
            observation, _ = env.reset()
            done = False
            while done == False:
                q_applied = q
                observation, reward, done, _, _ = env.step(q_applied)
                discounted_rewards += gamma ** steps * reward
                steps += 1
            Vq.append(discounted_rewards)
            Vqs[q, index] = discounted_rewards

    return INITIAL_STATES, Vqs



# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the environment
    env_kwargs = dict(TIME_STEPS=100,
                      initialize_state_randomly=True,
                      SAMPLING_TIME_SECONDS=0.05,
                      SET_POINTS=np.array([-1, 1], dtype=np.float32))
    env = SupervisorLineEnv(**env_kwargs)
    state_dim = env.observation_space.shape[0]
    gamma = 0.9

    env_kwargs_plotting = copy.deepcopy(env_kwargs)
    env_kwargs_plotting['initialize_state_randomly'] = False
    x_true, Vq_true = get_true_values(SupervisorLineEnv, env_kwargs=env_kwargs_plotting)
    
    resolution = 300
    neurons_set = [256]#[2,3,4,5,6,7,8] #[2, 4, 8, 16, 32, 64, 128, 256]#[2, 4, 6, 8, 10, 16, 32, 64]
    cmap = cm.viridis(np.linspace(0, 1, len(neurons_set)))  # Colormap for neuron configurations
    final_values_neurons = np.zeros((len(neurons_set), resolution), dtype=np.float32)
    for idneuron, n_neurons in enumerate(neurons_set):
        
        n_layers = 2
        activation_function = 'tanh'
        num_episodes = 5000  # Total number of episodes for training
        initial_lr = 0.001
        title = 'neurons: '+str(n_neurons)+', layers: '+str(n_layers) +', activation: '+activation_function + ', learnign rate:' +str(initial_lr).replace('.','_')
        # Define neural network architecture
        hidden_layers = [n_neurons] *n_layers # Number of neurons in hidden layers
        activation_functions = [activation_function]*n_layers  # Activation functions for each hidden layer
        
        fignumber = 1

        for policy_index in [0, 1]:
            save_name = 'agents/valuefunction' + str(policy_index) + '.pth'
            # Initialize the value network
            value_network = ValueNetwork(state_dim, hidden_layers, activation_functions)

            # Define optimizer

            optimizer = optim.Adam(value_network.parameters(), lr=initial_lr)

            # Define learning rate scheduler
            start_decay = int(0.75 * num_episodes)  # Start decay after 90% of epochs

            def lr_lambda(epoch):
                if epoch < start_decay:
                    return 1.0  # Keep the initial learning rate
                else:
                    # Linear decay to zero over the remaining epochs
                    return max(1e-5, (num_episodes - epoch) / (num_episodes - start_decay))

            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

            # Initialize experience replay buffer
            buffer_capacity = 10000  # Adjust based on your memory constraints
            replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        
            # Evaluate the policy
            evaluate_policy(env, save_name, policy_index,
                            value_network, optimizer, scheduler, replay_buffer,
                            gamma=gamma, num_episodes=num_episodes,
                            batch_size=64, return_threshold=0.1,
                            episode_interval=1000, get_state_values=get_state_values_lineenv,
                            use_multiple_starts=False)
            plt.figure(fignumber)
            plt.plot(x_true, Vq_true[policy_index,:], linewidth=3, color='black', 
                        label=rf'True value', linestyle='--')
            plt.title(title)
            plt.legend(loc='best')
            plt.show()
            fignumber += 10

            state_values = []
            for x in x_true:
                state = env.get_observations(x)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    value = value_network(state_tensor).item()
                state_values.append(value)
            final_values_neurons[idneuron, :] = np.array(state_values)
    
    plt.figure()
    average_error = []
    for idplot in range(len(neurons_set)):
        plt.plot(x_true, Vq_true[policy_index,:]-final_values_neurons[idplot,:], color=cmap[idplot], 
                label=str(neurons_set[idplot])+' Neurons')
    
    plt.xlabel('z', fontsize=22)
    plt.ylabel('$V-\hat{V}$', fontsize=22)
    plt.ylim(-0.1, 0.1)
    plt.legend(loc='best')
    plt.grid()

    plt.figure()
    
    for idplot in range(len(neurons_set)):
        plt.plot(x_true, Vq_true[policy_index,:]-final_values_neurons[idplot,:], color=cmap[idplot], 
                label=str(neurons_set[idplot])+' Neurons')
        average_error.append(np.mean(abs(Vq_true[policy_index,:]-final_values_neurons[idplot,:])))
    
    plt.xlabel('z', fontsize=22)
    plt.ylabel('$V-\hat{V}$', fontsize=22)
    plt.ylim(-0.01, 0.01)
    plt.legend(loc='best')
    plt.grid()

    plt.figure()
    plt.plot(neurons_set, average_error, '-o')
    plt.xlabel('z', fontsize=22)
    plt.ylabel('mean($|V-\hat{V}|$)', fontsize=22)
    plt.legend(loc='best')
    plt.grid()