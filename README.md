# Timer-Value-Supervisor

Simulation code for the paper **"A Timer-Based Hybrid Supervisor for Robust, Chatter-Free Policy Switching"**.

This project introduces a timer-based hybrid supervisor that prevents rapid, inefficient switching (chattering) between learned reinforcement learning policies in the presence of measurement noise. 

## Repository Structure

The code is organized into two main simulation environments:

* **`linetwopoints/`**: A 1D environment demonstrating the robust stabilization of two disconnected setpoints on a line. 
  * Core scripts: `line_env.py` (environment), `valueiteration.py` (value function training), and `test_timer_hybrid_alg.py` (supervisor simulation).
* **`multitarget/`**: A 2D planar environment featuring four disconnected setpoints with periodic reward variations to create complex switching boundaries.
  * Core scripts: `multitarget_env.py` (environment), `valueiteration.py` (value function training), and `test_timer_hybrid_alg.py` (supervisor simulation).

## Getting Started

This repository includes a `.devcontainer` configuration, which is the recommended way to run the simulations without manually managing dependencies.

### Using the Dev Container
1. Ensure you have [Docker](https://www.docker.com/) installed and running.
2. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
3. Clone this repository and open the folder in VS Code.
4. When prompted, click **Reopen in Container** (or open the Command Palette `Ctrl+Shift+P` and select `Dev Containers: Reopen in Container`). VS Code will automatically build the environment and install all requirements.

*(Alternatively, you can open this repository directly in **GitHub Codespaces** to run everything in your browser).*

### Running the Simulations
Once inside the container (or your local environment), navigate to either environment folder and run the testing scripts to execute the hybrid supervisor and reproduce the simulation plots.

**1D Two-Point Environment:**
```bash
cd linetwopoints
python test_timer_hybrid_alg.py
```
*(You can also visualize the value functions and policies by running `python plot_valuefunctionpolicy.py`)*

**2D Multi-Target Environment:**
```bash
cd multitarget
python test_timer_hybrid_alg.py
```
*(You can visualize the reward map by running `python plot_rewardmap.py`)*

## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@article{de_priester2025timer,
  title={A Timer-Based Hybrid Supervisor for Robust, Chatter-Free Policy Switching},
  author={de\_Priester, Jan and Sanfelice, Ricardo},
  journal={Reinforcement learning journal},
  volume={6},
  year={2025},
  publisher={Reinforcement Learning Journal}
}
```
