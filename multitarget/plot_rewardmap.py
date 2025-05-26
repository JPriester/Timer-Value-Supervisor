import numpy as np
import matplotlib.pyplot as plt

import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

def plot_reward(env):
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

    # Plot reward function
    plt.figure(figsize=(8, 6))
    c = plt.contourf(X, Y, reward_values, levels=20, cmap='viridis')
    cbar = plt.colorbar(c)
    cbar.set_label(label=r'$R$', fontsize=22)

    # Plot set points
    plt.scatter(env.SET_POINTS[:, 0], env.SET_POINTS[:, 1], color='red', marker='*', s=100, label="Setpoints")
    plt.legend(fontsize=16)
    plt.xlabel('$p_x$', fontsize=22)
    plt.ylabel('$p_y$', fontsize=22)
    # plt.title('Reward Function Heatmap', fontsize=16)
    plt.tight_layout()


if __name__ == "__main__":
    from multitarget_env import MultiTargetsEnv
    for distance_reward in [True, False]:
        plot_reward(MultiTargetsEnv(distance_reward=distance_reward))
        plt.savefig('plots/multitarg_reward_distreward_'+str(distance_reward)+'.pdf')