# Fallarm

Fallarm integrates [legged_gym](legged_gym/) and [rsl_rl](rsl_rl/) into a unified repository, providing Isaac Gym environments for legged robots together with a fast GPU-based PPO RL training framework.

**Maintainer**: Nikita Rudin  
**Affiliation**: Robotic Systems Lab, ETH Zurich & NVIDIA  
**Contact**: rudinn@ethz.ch  

---

## Components

### legged_gym — Isaac Gym Environments for Legged Robots

Provides environments used to train ANYmal (and other robots) to walk on rough terrain using NVIDIA's Isaac Gym. Includes all components needed for sim-to-real transfer: actuator network, friction & mass randomization, noisy observations and random pushes during training.

> :bell: **Announcement (09.01.2024):** With the shift from Isaac Gym to Isaac Sim at NVIDIA, the environments from this work have been migrated to [Isaac Lab](https://github.com/isaac-sim/IsaacLab). This repository will receive limited updates and support going forward.  
> Locomotion-related tasks in Isaac Lab are available [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html#locomotion).

Project website: https://leggedrobotics.github.io/legged_gym/  
Paper: https://arxiv.org/abs/2109.11978

---

### rsl_rl — Fast GPU-Based RL

Fast and simple implementation of RL algorithms designed to run fully on GPU. This code is an evolution of `rl-pytorch` provided with NVIDIA's Isaac GYM. Currently implements PPO; more algorithms may be added later.

---

## Installation

1. 创建conda环境
   ```bash
   conda create -n fallarm python=3.8 -y
   conda activate fallarm
   ```

2. 安装对应 cuda 12.1 的 pytorch 2.2.2【RTX 4090 可用】：
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    ```

3. 下载并安装[Isaac Gym](https://developer.nvidia.com/isaac-gym)：
    ```bash
    cd isaacgym/python && pip install -e .
    ```

4. 安装rsl_rl（PPO实现）和legged gym：
    ```bash
    cd rsl_rl && pip install -e . && cd .. 
    cd legged_gym &&  pip install -e . && cd .. 
    pip install tensorboard
    ```
---

## Code Structure

- `legged_gym/` — Isaac Gym legged robot environments
  - Each environment is defined by an env file (`legged_robot.py`) and a config file (`legged_robot_config.py`).
  - The config file contains two classes: environment parameters (`LeggedRobotCfg`) and training parameters (`LeggedRobotCfgPPo`).
  - Both env and config classes use inheritance.
  - Tasks must be registered using `task_registry.register(name, EnvClass, EnvConfig, TrainConfig)`.

- `rsl_rl/` — PPO RL training framework
  - Algorithms: `rsl_rl/algorithms/ppo.py`
  - Modules: `rsl_rl/modules/` (actor-critic, recurrent variants)
  - Runners: `rsl_rl/runners/on_policy_runner.py`
  - Storage: `rsl_rl/storage/rollout_storage.py`

---

## Usage

### Train
```bash
python legged_gym/scripts/train.py --task fall_arm --run_name test_fall_arm
```
- Run on CPU: add `--sim_device=cpu --rl_device=cpu`
- Run headless: add `--headless`
- **Tip**: Once training starts, press `v` to stop rendering for better performance.

Key CLI arguments:
| Argument | Description |
|---|---|
| `--task` | Task name |
| `--resume` | Resume from checkpoint |
| `--experiment_name` | Experiment name |
| `--run_name` | Run name |
| `--load_run` | Run to load when resuming (`-1` = last) |
| `--checkpoint` | Checkpoint number (`-1` = last) |
| `--num_envs` | Number of environments |
| `--seed` | Random seed |
| `--max_iterations` | Maximum training iterations |

### Play a Trained Policy
```bash
python legged_gym/scripts/play.py --task fall_arm --checkpoint_path /path/to/checkpoint.pt
```

---

## Adding a New Environment

1. Add a new folder to `legged_gym/envs/` with `<your_env>_config.py` inheriting from an existing config.
2. If adding a new robot:
   - Add assets to `legged_gym/resources/`.
   - Set asset path, body names, default joint positions and PD gains in `cfg`.
3. (Optional) Implement your environment in `<your_env>.py`, inheriting from an existing environment.
4. Register your env in `legged_gym/envs/__init__.py`.
5. Tune parameters in `cfg` and `cfg_train` as needed.

---

## Troubleshooting

- **`ImportError: libpython3.8m.so.1.0: cannot open shared object file`**  
  Run `sudo apt install libpython3.8`.  
  For conda: `export LD_LIBRARY_PATH=/path/to/conda/envs/<your_env>/lib`

---

## Known Issues

- Contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors attached to feet/end effectors only, excluding gravity:
  ```python
  sensor_options = gymapi.ForceSensorProperties()
  sensor_options.enable_forward_dynamics_forces = False
  sensor_options.enable_constraint_solver_forces = True
  sensor_options.use_world_frame = True
  ```

---

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.  
Third-party asset licenses: see `legged_gym/licenses/assets/`.  
Dependency licenses: see `legged_gym/licenses/dependencies/` and `rsl_rl/licenses/dependencies/`.
