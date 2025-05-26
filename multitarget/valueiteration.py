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
from multitarget_env import MultiTargetsEnv
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
                    batch_size=64,
                    episode_interval=25) :
    mse_loss = nn.MSELoss()
    max_episode_length = env._max_episode_steps

    # Lists to store losses and episodes
    losses = []
    episodes_list = []

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
                plot_value_function(env, value_network, resolution=100, policy_index=policy_index)
                plot_intervals.append(episode + 1)

    # Save the trained value network
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    torch.save(value_network.state_dict(), save_name)
    # Plot both Loss and Log Loss over Episodes
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



def get_true_values(Env, env_kwargs, resolution=100):
    # plot true value fun
    # Define grid resolution
    x_values = np.linspace(-1.5, 1.5, resolution)
    y_values = np.linspace(-1.5, 1.5, resolution)

    X, Y = np.meshgrid(x_values, y_values)
    Vqs = np.zeros((4, resolution, resolution), dtype=np.float32)
    for q in [0,1]:#, 2, 3]:
        Vq = []
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                env = Env(INITIAL_STATE=np.array([X[i, j], Y[i, j]]), **env_kwargs)

                discounted_rewards = 0
                steps = 0
                env.reset()
                observation, _ = env.reset()
                done = False
                while done == False:
                    q_applied = q
                    observation, reward, done, _, _ = env.step(q_applied)
                    discounted_rewards += gamma ** steps * reward
                    steps += 1
                Vq.append(discounted_rewards)
                Vqs[q, i, j] = discounted_rewards

    return X, Y, Vqs

def plot_value_function(env, value_network, resolution=100, policy_index=0):
    """
    Plots the estimated value function over the state space using the trained value network.

    Args:
        env: The environment instance.
        value_network: The trained neural network representing the value function.
        resolution: The number of points along each axis for evaluation.
        policy_index: The index of the policy used.
    """
    x_values = np.linspace(env.MINIMAL_VALUE_POSITION[0], env.MAXIMAL_VALUE_POSITION[0], resolution)
    y_values = np.linspace(env.MINIMAL_VALUE_POSITION[1], env.MAXIMAL_VALUE_POSITION[1], resolution)

    X, Y = np.meshgrid(x_values, y_values)
    V_estimated = np.zeros_like(X)

    # Evaluate the value function at each state
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                value = value_network(state_tensor).item()
            V_estimated[i, j] = value

    # Plot the estimated value function using pretty_plot
    plt.figure()
    c = plt.contourf(X, Y, V_estimated, levels=50, cmap='viridis')
    pretty_plot(c, f'$\\hat{{V}}_{{{policy_index+1}}}$')
    plt.show()


def pretty_plot(c, cbar_label):
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    cbar = plt.colorbar(c)
    cbar.set_label(label=cbar_label, fontsize=22)
    plt.grid()

# Main execution
if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    distance_reward = False

    # Define the environment
    env_kwargs = dict(TIME_STEPS=100, 
                    initialize_state_randomly=True, 
                    SAMPLING_TIME_SECONDS=0.025,  
                    SET_POINTS=np.array([[-1, 0.2], [-0.1, 1], [0.9, -.1], [-0.4, -0.9]], dtype=np.float32),
                    NOISE_MAGNITUDE = 0,
                    distance_reward=distance_reward)
    env = MultiTargetsEnv(**env_kwargs)
    state_dim = env.observation_space.shape[0]
    gamma = 0.9

    env_kwargs_plotting = copy.deepcopy(env_kwargs)
    env_kwargs_plotting['initialize_state_randomly'] = False
    x_true, y_true, Vq_true = get_true_values(MultiTargetsEnv, env_kwargs=env_kwargs_plotting)
    
    resolution = 100

        
    n_neurons = 1024
    n_layers = 2
    activation_function = 'tanh'
    num_episodes = 10000  # Total number of episodes for training
    initial_lr = 0.001
    title = 'neurons: '+str(n_neurons)+', layers: '+str(n_layers) +', activation: '+activation_function + ', learnign rate:' +str(initial_lr).replace('.','_')
    # Define neural network architecture
    hidden_layers = [n_neurons] *n_layers # Number of neurons in hidden layers
    activation_functions = [activation_function]*n_layers  # Activation functions for each hidden layer
    
    fignumber = 1

    for policy_index in [0, 1, 2, 3]:
        save_name = 'agents/valuefunction' + str(policy_index) + 'activation' + activation_function + 'neurons' + str(n_neurons)+ 'rewardmod_' + str(distance_reward) + '.pth'
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
                        batch_size=64, 
                        episode_interval=1000)
        # plt.legend(loc='best')
        plt.show()
        fignumber += 10

        state_values = np.zeros((x_true.shape[0], x_true.shape[1]))
        for i in range(x_true.shape[0]):
            for j in range(x_true.shape[1]):
                position = np.array([x_true[i,j], y_true[i,j]], dtype=np.float32)
                state = env.get_observations(position)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    value = value_network(state_tensor).item()
                state_values[i, j] = value
        final_values_numpy = np.array(state_values)
    
        plt.figure()
        c_true = plt.contourf(x_true, y_true, Vq_true[policy_index, :, :], levels=50, cmap='viridis')
        pretty_plot(c_true, f'$V_{policy_index+1}$')
        plt.figure()
        c_estimate = plt.contourf(x_true, y_true, final_values_numpy, levels=50, cmap='viridis')
        pretty_plot(c_estimate, f'$\hat{{V}}_{{{policy_index+1}}}$')
        plt.figure()
        c_estimate = plt.contourf(x_true, y_true, Vq_true[policy_index,:, :]-final_values_numpy, levels=50, cmap='viridis')
        pretty_plot(c_estimate, f'$V_{{{policy_index+1}}} - \hat{{V}}_{{{policy_index+1}}}$')


