import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # For color maps
from multitarget_env import MultiTargetsEnv
import matplotlib
from valueiteration import ValueNetwork
import matplotlib.colors as mcolors
from value_conditions import compute_value_maps

from matplotlib.lines import Line2D
from plot_rewardmap import plot_reward
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def load_value_function(save_name, state_dim, hidden_layers, activation_functions):
    value_network = ValueNetwork(state_dim, hidden_layers, activation_functions)
    state_dict = torch.load(save_name)
    value_network.load_state_dict(state_dict)
    return value_network


def plot_value_function(env, value_network, resolution=100, policy_index=0):
    x_values = np.linspace(env.MINIMAL_VALUE_POSITION[0], env.MAXIMAL_VALUE_POSITION[0], resolution)
    y_values = np.linspace(env.MINIMAL_VALUE_POSITION[1], env.MAXIMAL_VALUE_POSITION[1], resolution)
    X, Y = np.meshgrid(x_values, y_values)
    V_estimated = np.zeros_like(X)

    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j]])
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                value = value_network(state_tensor).item()
            V_estimated[i, j] = value

    plt.figure()
    c = plt.contourf(X, Y, V_estimated, levels=50, cmap='viridis')
    plt.scatter(env.SET_POINTS[:, 0], env.SET_POINTS[:, 1], color='red', marker='*', s=100, label='Set Points')
    pretty_plot(c, f'$\hat{{V}}_{{{policy_index+1}}}$')
    plt.show()



def pretty_plot(c, cbar_label):
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    cbar = plt.colorbar(c)
    cbar.set_label(label=cbar_label, fontsize=22)
    plt.grid()

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

def plot_reward_function_subplot(env, fign, ax_value):
    # Define grid resolution
    x_values = np.linspace(env.MINIMAL_VALUE_POSITION[0], env.MAXIMAL_VALUE_POSITION[0], 100)
    y_values = np.linspace(env.MINIMAL_VALUE_POSITION[1], env.MAXIMAL_VALUE_POSITION[1], 100)

    X, Y = np.meshgrid(x_values, y_values)
    reward_values = np.zeros_like(X)

    # Compute reward function for each point in the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            env.position = np.array([X[i, j], Y[i, j]])
            reward_values[i, j] = env.get_reward(env.position)
    ax_value.contourf(X, Y, reward_values, levels=20, cmap='viridis')
    # cbar = plt.colorbar(fig)
    # cbar.set_label(label=r'$R$', fontsize=22)
    ax_value.scatter(env.SET_POINTS[:, 0], env.SET_POINTS[:, 1], color='red', marker='*', s=100)

    ax_value.grid(visible=True)
    ax_value.set_xlabel('$p_x$', fontsize=22)
    ax_value.set_ylabel('$p_y$', fontsize=22)

def plot_max_q_regions_subplot(ax, X, Y, value_maps, setpoints):
    """
    Plot a map indicating which q-value (1,2,3,4) has the highest value at each region
    as a subplot on the given axis.

    Args:
        ax: Matplotlib axis object to plot on.
        X, Y: Meshgrid coordinates.
        value_maps: Dictionary containing value maps for each q (indexed from 0).
        setpoints: List of (x, y) setpoint coordinates.

    Returns:
        None (plots directly on the given axis).
    """
    # Stack value maps along a new axis and compute the argmax along that axis
    V_stack = np.stack([value_maps[q] for q in range(4)], axis=-1)
    q_max_map = np.argmax(V_stack, axis=-1) + 1  # Shift to make q in {1,2,3,4}

    # Define high-contrast colormap (No red)
    colors = ['#17becf', '#ff7f0e', '#2ca02c', '#7f7f7f']  # Cyan, Orange, Green, Dark Gray
    cmap = mcolors.ListedColormap(colors)

    # Set correct boundaries for color mapping
    bounds = np.array([0.5, 1.5, 2.5, 3.5, 4.5])  # Center each color on q values
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Use correct norm for pcolormesh
    mesh = ax.pcolormesh(X, Y, q_max_map, cmap=cmap, norm=norm, shading='auto')

    # Rasterize mesh for reduced file size
    mesh.set_rasterized(True)

    # Overlay setpoints in RED
    setpoints = np.array(setpoints)  # Ensure it's an array
    ax.scatter(setpoints[:, 0], setpoints[:, 1], color='red', marker='*', s=100)

    # Properly formatted color bar that matches the regions
    cbar = plt.colorbar(mesh, ax=ax, ticks=[1, 2, 3, 4])
    cbar.set_label(r"Optimal Policy Index $q$", fontsize=16)
    cbar.ax.set_yticklabels([r"$q=1$", r"$q=2$", r"$q=3$", r"$q=4$"])  # Ensure correct labeling

    # Labels and formatting
    ax.set_xlabel('$p_x$', fontsize=22)
    ax.set_ylabel('$p_y$', fontsize=22)
    ax.grid(True)


if __name__ == "__main__":
    distance_reward=False
    env_kwargs = dict(TIME_STEPS=120,#200,
                      SAMPLING_TIME_SECONDS=0.025,
                      SET_POINTS=np.array([[-1, 0.2], [-0.1, 1], [0.9, -.1], [-0.4, -0.9]], dtype=np.float32),
                      initialize_state_randomly=False,
                      distance_reward=distance_reward)
    env = MultiTargetsEnv(**env_kwargs)
    state_dim = env.observation_space.shape[0]

    hidden_layers = 2
    activation_function = 'tanh'
    n_neurons = 1024
    value_networks = []
    for policy_index in range(env.NUMBER_OF_qS):
        save_name = f'agents/valuefunction{policy_index}activation' + activation_function + 'neurons' + str(n_neurons) + 'rewardmod_' +  str(distance_reward) + '.pth'
        value_networks.append(load_value_function(save_name, 2, [n_neurons] * 2, [activation_function] * 2))
    resolution = 300
    x_values = np.linspace(-1.5, 1.5, resolution)
    y_values = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x_values, y_values)

    value_maps = {}

    for q in [0, 1, 2, 3]:
        value_maps[q] = compute_value_maps(value_networks[q], X, Y)  # Store the value function map

    init_conds = [
    (0.18241172, 0.11777468),   # Quadrant I (x > 0, y > 0), near origin
    (1.3965125, 1.214754),      # Quadrant I (x > 0, y > 0), extreme values
    (1.0215771, 0.89227504),    # Quadrant I (x > 0, y > 0), moderate values
    (0.9810862, -1.236564),     # Quadrant IV (x > 0, y < 0), moderate region
    (-1.4256433, -0.61712456),  # Quadrant III (x < 0, y < 0), extreme values
    (-0.9591057, -0.21514754),   # Quadrant III (x < 0, y < 0), moderate values
    (-1, 1.3),
    (-0.15, .2),
]

    NOISE_MAGNITUDE = 0*0.1
    mu = 0.15
    linecolors =  ['#17becf', '#ff7f0e', '#2ca02c', '#7f7f7f']
    modes = ['vanilla', 'hybrid', 'fixed']
    for idplot, mode in enumerate(modes):
        if mode == 'vanilla':
            search_horizon = 1
        else:
            search_horizon = 20
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))  # Create a 1x2 subplot layout
        ax_value, ax_time = axes[0], axes[1]

        plot_max_q_regions_subplot(ax_value, X, Y, value_maps, env.SET_POINTS)
        for init in init_conds:
            init = np.array(init, dtype=np.float32)
            
            xs = []
            ys = []
            q_maxes = []
            test_rewards = []

            test_done = False
            test_env = MultiTargetsEnv(**env_kwargs, INITIAL_STATE=init,
                                                NOISE_MAGNITUDE=NOISE_MAGNITUDE)
            obs, _ = test_env.reset()
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            # Initial value of q to use alongside the value of that q
            q_max, value_max = get_max_q_V(value_networks, obs_tensor)
            q_maxes.append(q_max)
            xs.append(test_env.position[0])
            ys.append(test_env.position[1])

            while test_done == False:
                search_env = MultiTargetsEnv(**env_kwargs, 
                            INITIAL_STATE=np.array(
                                                    test_env.position, 
                                                dtype=np.float32)) 
                search_env.reset()  
                for dwelltime_steps in range(search_horizon):
                    search_next_obs, _, _, _, _ = search_env.step(q_max)
                    search_next_obs_tensor = torch.from_numpy(search_next_obs).unsqueeze(0)
                    search_q_max, search_value_max = get_max_q_V(value_networks, search_next_obs_tensor)
                    current_value = value_networks[q_max](search_next_obs_tensor).item()
                    if mode == 'hybrid' and search_q_max != q_max and betterness_valuefunction(current_value, search_value_max, mu):#
                        # print(np.round(search_value_max,2), np.round(current_value,2))# and search_value_max/current_value > mu: # TODO how to implement the check properly
                        # print('better q available in' + str(dwelltime_steps) + ' steps at x=', np.round(search_env.x,2))
                        break
                # print('dwell time steps', dwelltime_steps)

                for _ in range(min(test_env.TIME_STEPS_left, dwelltime_steps+1)):
                    obs, test_reward, test_done, _, _ = test_env.step(q_max)
                    test_rewards.append(test_reward)
                    q_maxes.append(q_max)
                    xs.append(test_env.position[0])
                    ys.append(test_env.position[1])
                
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                q_max, value_max = get_max_q_V(value_networks, obs_tensor)
            

            # Overlay the solution trajectory on top of ax_value
            ax_value.plot(xs, ys, '--', color='black', linewidth=3, label='Solution Trajectory')
            ax_value.plot(xs[0], ys[0], 'o', color='black', linewidth=2, markersize=16, fillstyle='none', label='Start')
            ax_value.plot(xs[-1], ys[-1], 'x', color='black', linewidth=2, markersize=10, label='End')
            ax_value.scatter(env.SET_POINTS[:, 0], env.SET_POINTS[:, 1], color='red', marker='*', s=100, zorder = 10)

            timevec = np.linspace(0, len(xs)*test_env.SAMPLING_TIME_SECONDS, len(xs))

            for i in range(len(xs) - 1):  # Iterate over consecutive points
                segment_points = np.array([xs[i:i+2], ys[i:i+2]]).T  # Shape: (2, 2) → (N, 2)
    
                # Compute distance from segment points to each set point
                distances_to_targets = np.linalg.norm(segment_points[:, None, :] - env.SET_POINTS[None, :, :], axis=2)
                
                # Take the minimum distance to any set point
                distance_sp = np.min(distances_to_targets, axis=1)  # Shape: (2,)
                color = linecolors[q_maxes[i]] #if q_maxes[i] == 0 else linecolors[1]  # Choose color based on q_maxes
                ax_time.plot(timevec[i:i+2], distance_sp, color=color, linewidth=3)  # Plot segment with the chosen color
            # Compute the minimum distance of the start and end points to the set points
            start_distance = np.min(np.linalg.norm(np.array([xs[0], ys[0]]) - env.SET_POINTS, axis=1))
            end_distance = np.min(np.linalg.norm(np.array([xs[-1], ys[-1]]) - env.SET_POINTS, axis=1))
            if end_distance > 0.2:
                print('------------------')
                print('initial condition', init)
                print('end position error', end_distance)
            # Plot start and end markers
            # ax_time.plot(timevec[0], start_distance, 'o', color=color, linewidth=2, markersize=16, fillstyle='none', label="Start")
            # ax_time.plot(timevec[-1], end_distance, 'x', color=color, linewidth=2, markersize=10, label="End")

        ax_time.grid(visible=True)
        ax_time.set_xlabel('$t$', fontsize=22)
        ax_time.set_ylabel(r'$|p^*-p|$', fontsize=22)
        ax_time.set_ylim([0, 1.5])
        # Add legend for q lines
        legend_handles = [
            Line2D([0], [0], color=linecolors[0], lw=2, label='$q=1$'),
            Line2D([0], [0], color=linecolors[1], lw=2, label='$q=2$'),
            Line2D([0], [0], color=linecolors[2], lw=2, label='$q=3$'),
            Line2D([0], [0], color=linecolors[3], lw=2, label='$q=4$'),
        ]
        ax_time.legend(handles=legend_handles, fontsize=16)#, title="Legend for $q$")
        plt.tight_layout()
        plt.savefig('plots/multitargs_valuesims_noise'+str(NOISE_MAGNITUDE).replace('.', '')+'_stepsahead'+str(search_horizon)+'rewardmod_' + str(distance_reward) + '_' + mode+'.pdf')