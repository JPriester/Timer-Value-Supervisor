import torch
import torch.nn as nn
import numpy as np
from valueiteration import ValueNetwork
from line_env import SupervisorLineEnv
from test_timer_hybrid_alg import load_value_function
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def state_to_observation_tensor(x, setpoints=torch.tensor([-1, 1], requires_grad=False)):
    observation = x - setpoints
    return observation



# Define the network architecture (must match training)
state_dim = 2  # Replace with your actual state dimension
hidden_layers = [64]*2
activation_functions = ['relu']*2



env = SupervisorLineEnv(NOISE_MAGNITUDE=0)
# Load the saved model parameters
save_name0 = 'agents/valuefunction0.pth'
save_name1 = 'agents/valuefunction1.pth'
value_network0 = load_value_function(save_name0, state_dim, hidden_layers, activation_functions)
value_network1 = load_value_function(save_name1, state_dim, hidden_layers, activation_functions)
value_networks = [value_network0, value_network1]

# Define range of x values
x_values = torch.linspace(-3.0, 3.0, 100)  # Input range

linecolors = ['blue', 'green', 'orange', 'black']

plt.figure(1)
# plt.figure(2)
plt.figure(3)
plt.figure(4)

for index in [0,1]:
    # Compute value and gradients for each x
    gradients = []
    value_function_outputs = []
    for x in x_values:
        x = torch.tensor([x], requires_grad=True)  # Convert x to a leaf tensor
        observation = state_to_observation_tensor(x)
        value = value_networks[index](observation)  # Compute value (reshape x to match input dim)
        value.backward()  # Compute gradient
        gradients.append(x.grad.item())  # Store gradient
        value_function_outputs.append(value.item())  # Store value

    # Convert to NumPy for plotting
    x_np = x_values.numpy()
    value_np = np.array(value_function_outputs)
    grad_np = np.array(gradients)

    gradf_qmatch = grad_np*env.get_control_action(x_np, index)
    gradf_qnonmatch = grad_np*env.get_control_action(x_np, int(abs(index-1)))


    # Plot value function
    plt.figure(1)
    plt.plot(x_np, value_np, linewidth=3, color=linecolors[index], label="$q=$"+str(index+1))

    plt.figure(3)
    plt.plot(x_np, gradf_qmatch, linewidth=3, color=linecolors[index], label="$q=$"+str(index+1))
    
    plt.figure(4)
    plt.plot(x_np, gradf_qnonmatch, linewidth=3, color=linecolors[index], label="$q=$"+str(index+1))

plt.figure(1)
plt.grid(visible=True)
plt.plot([-1, 1], [0, 0], '*', color='red', markersize=12)
plt.xlabel('$z$', fontsize=22)
plt.ylabel(r'$V_q$', fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()

plt.figure(3)
plt.grid(visible=True)
plt.plot([-1, 1], [0, 0], '*', color='red', markersize=12)
plt.xlabel('$z$', fontsize=22)
plt.ylabel(r'$\langle\nabla V_q(z), f(z, \pi_{q}(z))\rangle$', fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()

plt.figure(4)
plt.grid(visible=True)
plt.plot([-1, 1], [0, 0], '*', color='red', markersize=12)
plt.xlabel('$z$', fontsize=22)
plt.ylabel(r'$\langle\nabla V_q(z), f(z, \pi_{q\prime}(z))\rangle$', fontsize=22)
plt.legend(fontsize=16)
plt.tight_layout()