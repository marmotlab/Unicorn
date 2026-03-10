<h1 align="center">
🦄 Unicorn: A Universal and Collaborative Reinforcement Learning Approach Toward Generalizable Network-Wide Traffic Signal Control
</h1>

<p align="center">
  <a href="https://ieeexplore.ieee.org/"><img src="https://img.shields.io/badge/📄_Paper-IEEE_T--ITS_2026-blue?style=for-the-badge" alt="Paper"></a>
  <a href="https://arxiv.org/abs/2503.11488"><img src="https://img.shields.io/badge/arXiv-2503.11488-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

<p align="center">
  <b>Official implementation of Unicorn, accepted in IEEE Transactions on Intelligent Transportation Systems (T-ITS).</b><br>
    <i>
    <a href="https://scholar.google.com/citations?user=o9zHCC0AAAAJ&hl=zh-CN">Yifeng Zhang</a>,
    <a href="#">Yilin Liu</a>,
    <a href="#">Ping Gong</a>,
    <a href="https://scholar.google.com/citations?user=kW4MON4AAAAJ&hl=zh-CN">Peizhuo Li</a>,
    <a href="https://scholar.google.com/citations?user=Oc7gaikAAAAJ&hl=zh-CN">Mingfeng Fan</a>,
    <a href="https://scholar.google.com/citations?user=n7NzZ0sAAAAJ&hl=zh-CN">Guillaume Sartoretti</a>
    </i><br>
  <a href="https://marmotlab.org/">MARMot Lab</a> @ National University of Singapore
</p>

<p align="center">
  <img src="images/framework.png" alt="Unicorn Framework" width="95%"/>
</p>

## 📋 Table of Contents
- [Highlights](#highlights)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Supported Datasets](#️supported-datasets)
- [Configuration](#️configuration)
- [Training](#training)
  - [Single-Scenario Training](#-single-scenario-training)
  - [Multi-Scenario Co-Training](#-multi-scenario-co-training)
- [Testing](#testing)
- [Evaluation with Non-RL Baselines](#evaluation-with-non-rl-baselines)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Unified Traffic Movement Representation**: A traffic movement-based state-action representation that unifies intersection states and signal phases across different intersection topologies.
- **Universal Traffic Representation (UTR) Module**: A decoder-only feature extraction architecture with cross-attention, designed to capture general traffic features across different intersections.
- **Intersection-Specific Representation (ISR) Module**: A feature extraction module combining a Variational Autoencoder (VAE) and contrastive learning to capture intersection-specific characteristics.
- **Collaborative Multi-Intersection Learning**: An attention-based coordination mechanism that adaptively models state-action dependencies among neighboring intersections for scalable network-level signal control.
- **Evaluation on Diverse Traffic Networks**: Experiments conducted on eight traffic datasets in SUMO, including three synthetic traffic networks and five real-world city-scale networks, supporting both single-scenario training and multi-scenario joint training.


---

## Requirements

| Dependency  | Version            |
|:------------|:-------------------|
| Python      | ≥ 3.8              |
| SUMO        | ≥ 1.16.0           |
| PyTorch     | 1.13.0 (CUDA 11.7) |
| Ray         | 2.3.1              |
| Gym         | 0.26.2             |
| SciPy       | 1.10.1             |
| einops      | 0.6.0              |
| NumPy       | 1.24.2             |
| TensorBoard | 2.13.0             |

> [!NOTE]
> Different PyTorch and CUDA versions may affect training performance and reproducibility. The code is tested with **PyTorch 1.13.0 + CUDA 11.7**.

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/marmotlab/Unicorn.git
cd Unicorn
```

### Step 2: Create Conda Environment

```bash
# Create a new conda environment
conda create -n unicorn python=3.8 -y
conda activate unicorn
```

### Step 3: Install PyTorch

Install PyTorch **first**, selecting the command that matches your CUDA version:

```bash
# CUDA 11.6
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

# CUDA 11.7
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

# CPU only
pip install torch==1.13.0+cpu torchvision==0.14.0+cpu torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

### Step 4: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Install SUMO Traffic Simulator

1. Download and install SUMO by following the official instructions at:
   👉 [https://sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php)

2. Set the environment variable `SUMO_HOME`. Refer to the [SUMO Basic Computer Skills guide](https://sumo.dlr.de/docs/Basics/Basic_Computer_Skills.html#linux) for detailed instructions on setting SUMO environment variables.

3. Verify the installation:
```bash
sumo --version
```

---

## Project Structure

```
Unicorn/
├── 📄 driver_unicorn.py         # Main training script (gradient updates & PPO optimization)
├── 📄 runner_unicorn.py         # Distributed experience collection via Ray workers
├── 📄 evaluator_rl.py           # Evaluation script for RL-based models (Unicorn)
├── 📄 evaluator_non_rl.py       # Evaluation script for non-RL baselines (Fixed, Greedy, Pressure)
├── 📄 parameters.py             # All training, simulation & experiment configurations
├── 📄 utils.py                  # Utility functions
├── 📄 requirements.txt          # Python dependencies
│
├── 📂 models/
│   └── Unicorn.py               # Unicorn network architecture (Actor-Critic)
│
├── 📂 env/
│   ├── matsc.py                 # Multi-Agent TSC Gym environment (SUMO interface)
│   └── tls.py                   # Traffic light signal controller module
│
├── 📂 maps/                     # SUMO network datasets & configuration files
│   ├── grid_network_5_5/        # Synthetic 5×5 Grid (MA2C)
│   ├── monaco_network_30/       # Real-world Monaco 30 intersections (MA2C)
│   ├── cologne_network_8/       # Real-world Cologne 8 intersections (RESCO)
│   ├── ingolstadt_network_21/   # Real-world Ingolstadt 21 intersections (RESCO)
│   ├── grid_network_4_4/        # Synthetic 4×4 Grid (RESCO)
│   ├── arterial_network_4_4/    # Synthetic 4×4 Arterial (RESCO)
│   ├── shaoxing_network_7/      # Real-world Shaoxing 7 intersections (GESA)
│   ├── shenzhen_network_29/     # Real-world Shenzhen 29 intersections (GESA)
│   └── shenzhen_network_55/     # Real-world Shenzhen 55 intersections (GESA)
│
└── 📂 images/
    └── framework.png            # Framework overview figure
```

---

## Supported Datasets

Unicorn is evaluated on **8 SUMO traffic network scenarios** from three benchmark suites:

| Benchmark | Network | # Intersections | Type |
|:---------:|:--------|:---------------:|:----:|
| **[MA2C](https://github.com/cts198859/deeprl_signal_control)** | `grid_network_5_5` | 25 | Synthetic |
| **[MA2C](https://github.com/cts198859/deeprl_signal_control)** | `monaco_network_30` | 30 | Real-world |
| **[RESCO](https://github.com/Pi-Star-Lab/RESCO)** | `cologne_network_8` | 8 | Real-world |
| **[RESCO](https://github.com/Pi-Star-Lab/RESCO)** | `ingolstadt_network_21` | 21 | Real-world |
| **[RESCO](https://github.com/Pi-Star-Lab/RESCO)** | `grid_network_4_4` | 16 | Synthetic |
| **[RESCO](https://github.com/Pi-Star-Lab/RESCO)** | `arterial_network_4_4` | 16 | Synthetic |
| **[GESA](https://github.com/AnonymousIDforSubmission/GESA)** | `shaoxing_network_7` | 7 | Real-world |
| **[GESA](https://github.com/AnonymousIDforSubmission/GESA)** | `shenzhen_network_29` | 29 | Real-world |

> **References:**
> - **[MA2C](https://ieeexplore.ieee.org/document/8667868)**: Chu, T., Wang, J., Codecà, L., & Li, Z. (2020). *Multi-Agent Deep Reinforcement Learning for Large-Scale Traffic Signal Control.* IEEE T-ITS.
> - **[RESCO](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/f0935e4cd5920aa6c7c996a5ee53a70f-Paper-round1.pdf)**: Ault, J., & Sharon, G. (2021). *Reinforcement Learning Benchmarks for Traffic Signal Control.* NeurIPS Datasets & Benchmarks.
> - **[GESA](https://ieeexplore.ieee.org/document/10481508)**: Jiang, H., et al. (2024). *A General Scenario-Agnostic Reinforcement Learning for Traffic Signal Control.* IEEE T-ITS.

---

## ⚙️ Configuration

All configurations are centralized in [`parameters.py`](parameters.py). Below are the key parameters:

### 🔹 Input Parameters (`INPUT_PARAMS`)

| Parameter | Description | Default |
|:----------|:------------|:--------|
| `MAX_EPISODES` | Total number of training episodes | `3000` |
| `NUM_META_AGENTS` | Number of parallel Ray worker processes | `6` |
| `LOAD_MODEL` | Whether to resume training from a checkpoint | `False` |
| `EXPERIMENT_PATH` | Path to the checkpoint experiment (for resume) | `None` |
| `CO_TRAIN` | **Enable multi-scenario co-training** | `False` |

> [!IMPORTANT]
> When switching between single-scenario and multi-scenario training modes, make sure to set `CO_TRAIN` accordingly in `INPUT_PARAMS` and configure the appropriate dataset(s).

---

## Training

### 🔸 Single-Scenario Training

In single-scenario mode, the model trains on **one specific traffic network** at a time. This is the default mode.

**Step 1:** Configure [`parameters.py`](parameters.py):

```python
class INPUT_PARAMS:
    MAX_EPISODES    = 3000        # Total training episodes
    NUM_META_AGENTS = 6           # Number of parallel workers
    CO_TRAIN        = False       # ⬅️ Set to False for single-scenario

class SUMO_PARAMS:
    NET_NAME        = 'grid_network_5_5'  # ⬅️ Choose your target dataset
```

**Step 2:** Launch training:

```bash
python driver_unicorn.py
```

> [!TIP]
> **Recommended configurations by dataset:**
>
| Dataset | Green / Yellow | Teleport Time |
|:--------|:--------------:|:-------------:|
| MA2C networks | 10s / 3s | 300 |
| RESCO networks | 15s / 5s | -1 |
| GESA networks | 15s / 5s | 600 |

### 🔸 Multi-Scenario Co-Training

In multi-scenario co-training mode, the model trains **simultaneously across multiple traffic networks** using different workers for different scenarios. This enables cross-domain generalization.

**Step 1:** Configure [`parameters.py`](parameters.py):

```python
class INPUT_PARAMS:
    MAX_EPISODES    = 3000
    NUM_META_AGENTS = 6           # Each worker trains on a different scenario
    CO_TRAIN        = True        # ⬅️ Set to True for multi-scenario

class SUMO_PARAMS:
    ALL_DATASETS    = [           # ⬅️ Define the scenarios to co-train on
        'cologne_network_8',
        'ingolstadt_network_21',
        'arterial_network_4_4',
        'grid_network_4_4',
        'shaoxing_network_7',
        'shenzhen_network_29',
    ]
```

> [!NOTE]
> In co-training mode:
> - Each worker (indexed by `server_number`) is assigned a different dataset from `ALL_DATASETS`.
> - The number of `NUM_META_AGENTS` should match the number of datasets in `ALL_DATASETS`.
> - The observation and action spaces are automatically padded to the maximum dimensions across all scenarios (max movement dim = 36, max phase dim = 8, agent space = 97).

**Step 2:** Launch training:

```bash
python driver_unicorn.py
```

### Monitoring Training

Training logs are automatically recorded with [TensorBoard](https://www.tensorflow.org/tensorboard):

```bash
tensorboard --logdir ./Train_MATSC/<EXPERIMENT_NAME>/train
```

Key metrics tracked:
- **Policy Loss, Value Loss, Entropy Loss**
- **Actor/Critic VAE Loss & Contrastive Loss**
- **Episode Reward, Episode Length, Action Change Rate**

### Resume Training from Checkpoint

To resume training from a saved checkpoint:

```python
class INPUT_PARAMS:
    LOAD_MODEL      = True
    EXPERIMENT_PATH = './Train_MATSC/<YOUR_EXPERIMENT_NAME>'  # ⬅️ Path to saved experiment
```

---

## Testing

After training, evaluate the trained model on specific test scenarios.

**Step 1:** Configure the test settings in [`evaluator_rl.py`](evaluator_rl.py):

```python
# Set the experiment directory and model path
exp_dir = './Test'
agent_name_list = ['UNICORN']
model_path_list = ['./Train_MATSC/<EXPERIMENT_NAME>/model/checkpoint<EPISODE>.pkl']
```

**Step 2:** Ensure the corresponding map and flow settings in [`parameters.py`](parameters.py) match the training configuration:

```python
class SUMO_PARAMS:
    NET_NAME = 'grid_network_5_5'  # ⬅️ Must match the map used during training
```

**Step 3:** Run evaluation:

```bash
python evaluator_rl.py
```

**Step 4:** After testing, the results (traffic data & trip data) will be saved in:
```
./Test/eval_data/
├── <map_name>_UNICORN_traffic.csv    # Traffic metrics per timestep
└── <map_name>_UNICORN_trip.csv       # Individual vehicle trip info
```

---

## Evaluation with Non-RL Baselines

Unicorn includes built-in non-RL baseline evaluators for comparison:

| Baseline | Description |
|:---------|:------------|
| `FIXED` | Fixed-time signal plan |
| `GREEDY` | Greedy policy based on queue length |
| `PRESSURE` | Max-pressure based control |

**Run baseline evaluations:**

```bash
python evaluator_non_rl.py
```

Configure the baselines in [`evaluator_non_rl.py`](evaluator_non_rl.py):
```python
agent_name_list = ['FIXED', 'GREEDY', 'PRESSURE']
```

---

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@ARTICLE{11360985,
  author={Zhang, Yifeng and Liu, Yilin and Gong, Ping and Li, Peizhuo and Fan, Mingfeng and Sartoretti, Guillaume},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Unicorn: A Universal and Collaborative Reinforcement Learning Approach Toward Generalizable Network-Wide Traffic Signal Control}, 
  year={2026},
  volume={},
  number={},
  pages={1-17},
  keywords={Collaboration;Topology;Network topology;Feature extraction;Vectors;Urban areas;Real-time systems;Training;Reinforcement learning;Scalability;Generalizable adaptive traffic signal control;multi-agent reinforcement learning;contrastive learning},
  doi={10.1109/TITS.2026.3653478}}

```

You may also find our related work useful:

```bibtex
@INPROCEEDINGS{10801524,
  author={Zhang, Yifeng and Li, Peizhuo and Fan, Mingfeng and Sartoretti, Guillaume},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={HeteroLight: A General and Efficient Learning Approach for Heterogeneous Traffic Signal Control}, 
  year={2024},
  volume={},
  number={},
  pages={1010-1017},
  keywords={Measurement;Network topology;Urban areas;Reinforcement learning;Feature extraction;Vectors;Robustness;Topology;Optimization;Intelligent robots},
  doi={10.1109/IROS58592.2024.10801524}}

```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

© 2026 [MARMot Lab](https://marmotlab.org/) @ NUS-ME

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

</div>
