import torch
import torch.nn as nn
import numpy as np
from valueiteration import ValueNetwork
from line_env import SupervisorLineEnv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def load_value_function(save_name, state_dim, hidden_layers, activation_functions):
    # Initialize the network
    value_network = ValueNetwork(state_dim, hidden_layers, activation_functions)
    state_dict = torch.load(save_name)
    value_network.load_state_dict(state_dict)
    return value_network

def plot_value_functions(value_networks, env, fign, linecolors):
    plt.figure(fign)

    x_values = np.linspace(env.MINIMAL_VALUE_POSITION_X, env.MAXIMAL_VALUE_POSITION_X, 200)
    with torch.no_grad():
        for q, V in enumerate(value_networks):
            label=rf'$q = {q+1}$'
            state_values = []
            for x in x_values:
                observation = env.get_observations(x)
                state_tensor = torch.from_numpy(observation).unsqueeze(0)
                value = V(state_tensor)
                state_values.append(value.item())
            plt.plot(x_values, state_values,  linewidth=3, color=linecolors[q], 
                    label=label)
    plt.grid(visible=True)
    plt.plot(env.SET_POINTS, [0, 0], '*', color='red', markersize=12)
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel(r'$V_q$', fontsize=22)
    plt.legend(fontsize=16)

    plt.tight_layout()

def plot_value_functions_subplot(value_networks, env, fign, linecolors, ax_value):
    x_values = np.linspace(env.MINIMAL_VALUE_POSITION_X, env.MAXIMAL_VALUE_POSITION_X, 200)
    with torch.no_grad():
        for q, V in enumerate(value_networks):
            label=rf'$q = {q+1}$'
            state_values = []
            for x in x_values:
                observation = env.get_observations(x)
                state_tensor = torch.from_numpy(observation).unsqueeze(0)
                value = V(state_tensor)
                state_values.append(value.item())
            ax_value.plot(x_values, state_values,  linewidth=3, color=linecolors[q], 
                    label=label)
    ax_value.grid(visible=True)
    ax_value.plot(env.SET_POINTS, [0, 0], '*', color='red', markersize=12)
    ax_value.set_xlabel('$z$', fontsize=22)
    ax_value.set_ylabel(r'$V_q$', fontsize=22)
    ax_value.legend(fontsize=16)


def betterness_valuefunction(V_old, V_new, mu, epsilon=1e-6):
    abs_V_old = abs(V_old)
    if abs_V_old <= epsilon:
        return V_new > epsilon
    signed_ratio = (V_new - V_old) / (abs_V_old + epsilon)
    # print('signed_ratio', signed_ratio)
    return signed_ratio >= mu

def get_max_q_V(value_networks, obs_tensor):
    state_values = []
    with torch.no_grad():
        for _, V in enumerate(value_networks):
            state_values.append(V(obs_tensor).item())
    q_max = np.argmax(np.array(state_values))
    value_max = state_values[q_max]
    return q_max, value_max

# Main execution
if __name__ == "__main__":
    # Define the network architecture (must match training)
    state_dim = 2  # Replace with your actual state dimension
    hidden_layers = [256]*2
    activation_functions = ['tanh', 'tanh']

    # Load the saved model parameters
    save_name0 = 'agents/valuefunction0.pth'
    save_name1 = 'agents/valuefunction1.pth'
    value_network0 = load_value_function(save_name0, state_dim, hidden_layers, activation_functions)
    value_network1 = load_value_function(save_name1, state_dim, hidden_layers, activation_functions)
    value_networks = [value_network0, value_network1]


    # Define the environment
    env_kwargs = dict(TIME_STEPS=100,
                SAMPLING_TIME_SECONDS=0.05,
                SET_POINTS=np.array([-1, 1], dtype=np.float32),
                initialize_state_randomly=False)

    env = SupervisorLineEnv(**env_kwargs)
    linecolors = ['blue', 'green', 'red', 'orange', 'black']


    init_conds = [-3, -0.1, 0.1, 2.5]
    NOISE_MAGNITUDE = 0.3
    mu = 0.1
    
    for idplot, search_horizon in enumerate([1, 10]):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))  # Create a 1x2 subplot layout
        ax_value, ax_time = axes[0], axes[1]

        plot_value_functions_subplot(value_networks, env, idplot, linecolors, ax_value)
        for x in init_conds:
            xs = []
            q_maxes = []
            test_rewards = []

            test_done = False
            test_env = SupervisorLineEnv(**env_kwargs, INITIAL_STATE=np.array(x, dtype=np.float32),
                                                NOISE_MAGNITUDE=NOISE_MAGNITUDE)
            obs, _ = test_env.reset()
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # Initial value of q to use alongside the value of that q
            q_max, value_max = get_max_q_V(value_networks, obs_tensor)
            q_maxes.append(q_max)
            xs.append(test_env.x)

            while test_done == False:
                search_env = SupervisorLineEnv(**env_kwargs, 
                            INITIAL_STATE=np.array(
                                                    test_env.x, 
                                                dtype=np.float32)) 
                search_env.reset()  
                for dwelltime_steps in range(search_horizon):
                    search_next_obs, _, _, _, _ = search_env.step(q_max)
                    search_next_obs_tensor = torch.from_numpy(search_next_obs).unsqueeze(0)
                    search_q_max, search_value_max = get_max_q_V(value_networks, search_next_obs_tensor)
                    current_value = value_networks[q_max](search_next_obs_tensor).item()
                    if search_q_max != q_max and betterness_valuefunction(current_value, search_value_max, mu):#
                        # print(np.round(search_value_max,2), np.round(current_value,2))# and search_value_max/current_value > mu: # TODO how to implement the check properly
                        # print('better q available in' + str(dwelltime_steps) + ' steps at x=', np.round(search_env.x,2))
                        break
                # print('dwell time steps', dwelltime_steps)

                for _ in range(min(test_env.TIME_STEPS_left, dwelltime_steps+1)):
                    obs, test_reward, test_done, _, _ = test_env.step(q_max)
                    test_rewards.append(test_reward)
                    q_maxes.append(q_max)
                    xs.append(test_env.x)
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                q_max, value_max = get_max_q_V(value_networks, obs_tensor)
            value_trajectory = []
            for idx, xpoint in enumerate(xs):
                obs_point = test_env.get_observations(xpoint)
                obs_point_tensor = torch.from_numpy(obs_point).unsqueeze(0)
                value_trajectory.append(value_networks[q_maxes[idx]](obs_point_tensor).item())
            
            # plt.figure(idplot)
            ax_value.plot(xs, value_trajectory, '--' , color=linecolors[2], linewidth=3)
            ax_value.plot(xs[0], value_trajectory[0], 'o', color=linecolors[2], linewidth=2, markersize=16, fillstyle='none')
            ax_value.plot(xs[-1], value_trajectory[-1], 'x', color=linecolors[2], linewidth=2, markersize=10)

            # plt.figure(idplot+100)
            timevec = np.linspace(0, len(xs)*test_env.SAMPLING_TIME_SECONDS, len(xs))
            plt.plot(timevec, xs)
            for i in range(len(xs) - 1):  # Iterate over consecutive points
                color = linecolors[q_maxes[i]] #if q_maxes[i] == 0 else linecolors[1]  # Choose color based on q_maxes
                ax_time.plot(timevec[i:i+2], xs[i:i+2], color=color, linewidth=2)  # Plot segment with the chosen color
            ax_time.plot(timevec[0], xs[0], 'o', color=linecolors[q_maxes[0]], linewidth=2, markersize=16, fillstyle='none')
            ax_time.plot(timevec[-1], xs[-1], 'x', color=linecolors[q_maxes[-1]], linewidth=2, markersize=10)

        # plt.figure(idplot+100)
        ax_time.plot([timevec[0], timevec[-1]], [test_env.SET_POINTS[0]]*2, '--', color='red')
        ax_time.plot([timevec[0], timevec[-1]], [test_env.SET_POINTS[1]]*2, '--', color='red')
        ax_time.grid(visible=True)
        ax_time.set_xlabel('$t$', fontsize=22)
        ax_time.set_ylabel(r'$z$', fontsize=22)
        ax_time.set_ylim([-3, 3])
        # Add legend for q lines
        legend_handles = [
            Line2D([0], [0], color=linecolors[0], lw=2, label='$q=1$'),
            Line2D([0], [0], color=linecolors[1], lw=2, label='$q=2$'),
        ]
        ax_time.legend(handles=legend_handles, fontsize=16)#, title="Legend for $q$")
        plt.tight_layout()

        plt.savefig('plots/twoline_valuesims_noise'+str(NOISE_MAGNITUDE).replace('.', '')+'_stepsahead'+str(search_horizon)+'.pdf')



