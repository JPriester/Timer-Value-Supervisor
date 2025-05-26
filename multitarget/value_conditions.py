import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from multitarget_env import MultiTargetsEnv
from valueiteration import ValueNetwork
import matplotlib
import matplotlib.colors as mcolors

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def load_value_function(save_name, state_dim, hidden_layers, activation_functions):
    """Load a trained value function network."""
    value_network = ValueNetwork(state_dim, hidden_layers, activation_functions)
    state_dict = torch.load(save_name)
    value_network.load_state_dict(state_dict)
    return value_network


def compute_value_maps(value_network, x_values, y_values):
    """
    Compute the value function over a 2D grid.

    Args:
        value_network: The trained neural network representing the value function.
        x_values, y_values: The 2D grid points for evaluation.

    Returns:
        Value function map over the 2D grid.
    """
    V = np.zeros_like(x_values)

    for i in range(x_values.shape[0]):
        for j in range(y_values.shape[1]):
            state = torch.tensor([x_values[i, j], y_values[i, j]], dtype=torch.float32)
            state_tensor = state.unsqueeze(0)  # Make it batch format
            with torch.no_grad():
                value = value_network(state_tensor)  # Compute value function output
            V[i, j] = value.item()

    return V


def compute_value_gradients(value_network, x_values, y_values):
    """
    Compute the gradient of the value function over a 2D grid.

    Returns:
        Gradients (dV/dx, dV/dy) over the 2D grid.
    """
    grad_x = np.zeros_like(x_values)
    grad_y = np.zeros_like(y_values)

    for i in range(x_values.shape[0]):
        for j in range(y_values.shape[1]):
            state = torch.tensor([x_values[i, j], y_values[i, j]], dtype=torch.float32, requires_grad=True)
            state_tensor = state.unsqueeze(0)  # Make it batch format
            value = value_network(state_tensor)  # Compute value function output
            value.backward()  # Compute gradient

            grad_x[i, j] = state.grad[0].item()
            grad_y[i, j] = state.grad[1].item()

    return grad_x, grad_y


def compute_gradient_inner_products(env, grad_x, grad_y, x_values, y_values, policy_index):
    """
    Compute inner products of the value function gradient with control actions.

    Returns:
        Inner product results: <∇V_q, f(x, y, π_q)> and <∇V_q, f(x, y, π_q')>.
    """
    gradf_qmatch = np.zeros_like(grad_x)
    gradf_qnonmatch = {q: np.zeros_like(grad_x) for q in [0, 1, 2, 3] if q != policy_index}

    for i in range(x_values.shape[0]):
        for j in range(y_values.shape[1]):
            position = np.array([x_values[i, j], y_values[i, j]], dtype=np.float32)

            # Get the control action for the selected policy
            control_q = env.get_control_action(position, policy_index)
            gradf_qmatch[i, j] = grad_x[i, j] * control_q[0] + grad_y[i, j] * control_q[1]

            # Compute for all other policies q'
            for q_other in gradf_qnonmatch.keys():
                control_q_other = env.get_control_action(position, q_other)
                gradf_qnonmatch[q_other][i, j] = grad_x[i, j] * control_q_other[0] + grad_y[i, j] * control_q_other[1]

    return gradf_qmatch, gradf_qnonmatch


def plot_contour(X, Y, Z, title, cbar_label):
    """Helper function to plot contour maps."""
    plt.figure(figsize=(7, 6))
    c = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(c, label=cbar_label)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    plt.title(title, fontsize=22)
    plt.grid()
    plt.show()

def plot_max_q_regions(X, Y, value_maps, setpoints):
    """
    Plot a map indicating which q-value (1,2,3,4) has the highest value at each region,
    and overlay setpoints in red.

    Args:
        X, Y: Meshgrid coordinates.
        value_maps: Dictionary containing value maps for each q (indexed from 0).
        setpoints: List of (x, y) setpoint coordinates.

    Returns:
        None (displays a plot).
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

    plt.figure()
    
    # Use correct norm for pcolormesh
    mesh = plt.pcolormesh(X, Y, q_max_map, cmap=cmap, norm=norm, shading='auto')

    mesh.set_rasterized(True) 
    # Overlay setpoints in RED
    setpoints = np.array(setpoints)  # Ensure it's an array
    plt.scatter(setpoints[:, 0], setpoints[:, 1], color='red', marker='*', s=100, label="Setpoints")

    # Properly formatted color bar that matches the regions
    cbar = plt.colorbar(mesh, ticks=[1, 2, 3, 4])
    # cbar.set_label(r"Optimal Policy Index $q$", fontsize=16)
    cbar.ax.set_yticklabels([r"$q=1$", r"$q=2$", r"$q=3$", r"$q=4$"])  # Ensure correct labeling

    # Labels and title
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    # plt.title("Regions of Optimal Policy $q$", fontsize=22)
    plt.legend(fontsize=16)
    plt.grid()
    plt.tight_layout()



# ---------------- MAIN EXECUTION ---------------- #
if __name__ == "__main__":
    plot_all = False
    # Define network parameters
    state_dim = 2
    n_neurons  = 256
    hidden_layers = [n_neurons] * 2
    activation_function =  'relu'
    activation_functions = [activation_function] * 2
    distance_reward = False
    

    # Initialize the environment
    env = MultiTargetsEnv(NOISE_MAGNITUDE=0)

    # Define the 2D grid for state space evaluation
    resolution = 300
    x_values = np.linspace(-1.5, 1.5, resolution)
    y_values = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x_values, y_values)

    # Load all value networks
    value_networks = []
    value_maps = {}

    for q in [0, 1, 2, 3]:
        save_name = f'agents/valuefunction{q}activation' + activation_function + 'neurons' + str(n_neurons)+'rewardmod_' + str(distance_reward) + '.pth'
        value_networks.append(load_value_function(save_name, state_dim, hidden_layers, activation_functions))
        value_maps[q] = compute_value_maps(value_networks[-1], X, Y)  # Store the value function map

    if plot_all:
        # Plot absolute difference between each value function map
        for q1 in range(4):
            for q2 in range(q1 + 1, 4):  # Only consider pairs (q1, q2) where q1 < q2
                diff_map = np.abs(value_maps[q1] - value_maps[q2])
                plot_contour(X, Y, diff_map, title=f"Absolute Difference |V_{q1} - V_{q2}|",
                            cbar_label=r"$|V_{q_1} - V_{q_2}|$")

        # Iterate over each policy q
        for policy_index in [0, 1, 2, 3]:
            print(f"Processing Policy {policy_index+1}")

            # Compute Value Function Gradients
            grad_x, grad_y = compute_value_gradients(value_networks[policy_index], X, Y)

            # Compute Inner Products of Gradients with f(x, y, π_q) and f(x, y, π_q')
            gradf_qmatch, gradf_qnonmatch = compute_gradient_inner_products(env, grad_x, grad_y, X, Y, policy_index)

            # Plot the Value Function Gradient Norm
            grad_norm = np.sqrt(grad_x ** 2 + grad_y ** 2)
            plot_contour(X, Y, grad_norm, title=f"Gradient Norm |∇V_{policy_index+1}|", cbar_label=r"$|\nabla V_q|$")

            # Plot the Inner Product <∇V_q, f(x, y, π_q)>
            plot_contour(X, Y, gradf_qmatch, title=f"Gradient Alignment ⟨∇V_{policy_index+1}, f(x, y, π_{policy_index+1})⟩",
                        cbar_label=r"$\langle\nabla V_q, f(x, y, \pi_q)\rangle$")

            # Plot the Inner Product <∇V_q, f(x, y, π_q')> for all other q'
            for q_other, grad_q_other in gradf_qnonmatch.items():
                plot_contour(X, Y, grad_q_other,
                            title=f"Cross Gradient Alignment ⟨∇V_{policy_index+1}, f(x, y, π_{q_other})⟩",
                            cbar_label=r"$\langle\nabla V_q, f(x, y, \pi_{q\prime})\rangle$")
    
    # Call the function to plot the region map
    plot_max_q_regions(X, Y, value_maps, env.SET_POINTS)
    plt.savefig(f'plots/multitargs_qmax_rewarddist'+ str(distance_reward) + '.pdf')