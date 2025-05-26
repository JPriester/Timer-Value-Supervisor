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


def betterness_valuefunction(V_old, V_new, mu, epsilon=1e-6):
    abs_V_old = abs(V_old)
    if abs_V_old <= epsilon:
        return V_new > epsilon
    signed_ratio = (V_new - V_old) / (abs_V_old + epsilon)
    # print('signed_ratio', signed_ratio)
    return signed_ratio >= mu - 1

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


    x_range = np.linspace(-3, 3, 300)
    
    plot_value_functions(value_networks, env, 1, linecolors)
    plt.savefig('plots/twoline_valuefunction.pdf')
    
    q_maxes = []
    controls = []
    for x in x_range:

        test_env = SupervisorLineEnv(**env_kwargs, INITIAL_STATE=np.array(x, dtype=np.float32),
                                            )
        obs, _ = test_env.reset()
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        # Initial value of q to use alongside the value of that q
        q_max, value_max = get_max_q_V(value_networks, obs_tensor)
        q_maxes.append(q_max)
        controls.append(env.get_control_action(x, q_max))

    
    plt.figure(2)
    for i in range(len(x_range) - 1):  # Iterate over consecutive points
        color = linecolors[q_maxes[i]]  # Choose color based on q_maxes
        if abs(controls[i+1] - controls[i]) <= 0.1:  # Skip if difference is too big
            plt.plot(x_range[i:i+2], controls[i:i+2], color=color, linewidth=3)  # Plot segment
        else:
            print('z = ', x_range[i:i+2])
    # plt.figure(idplot+100)
    plt.grid(visible=True)
    plt.xlabel('$z$', fontsize=22)
    plt.ylabel(r'$u$', fontsize=22)
    # plt.ylim([-3, 3])
    # Add legend for q lines
    legend_handles = [
        Line2D([0], [0], color=linecolors[0], lw=2, label='$q=1$'),
        Line2D([0], [0], color=linecolors[1], lw=2, label='$q=2$'),
    ]
    plt.legend(handles=legend_handles, fontsize=16)#, title="Legend for $q$")
    plt.tight_layout()
    plt.savefig('plots/twoline_policy.pdf')
    # plt.savefig('plots/twoline_timesims_noise'+str(NOISE_MAGNITUDE).replace('.', '')+'_stepsahead'+str(search_horizon)+'.pdf')
    
    # plt.title('hybrid logic implemented, noise = ' +str(NOISE_MAGNITUDE)+', $\mu=$'+str(mu)+', search '+str(search_horizon))
    # plt.figure(idplot)

    # plt.title('hybrid logic implemented, noise = ' +str(NOISE_MAGNITUDE)+', $\mu=$'+str(mu)+', search '+str(search_horizon))
        
    # Try with noise



