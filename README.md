# DRLND
Deep Reinforcement Learning Nanodegree from Udacity. The original course [GitHub repository](https://github.com/udacity/deep-reinforcement-learning).

## Contents
### Assignments
#### Part 1: Introduction to Deep Reinforcement Learning
 - Lesson 6: Monte Carlo Methods
    - [OpenAI Gym BlackJackEnv](./assignments/P1-Intro/L6-Monte-Carlo/Monte_Carlo.ipynb)
 - Lesson 7: Temporal-Difference Methods
    - [OpenAI Gym CliffWalkingEnv](./assignments/P1-Intro/L7-Temporal-Difference/Temporal_Difference.ipynb)
 - Lesson 8: OpenAI Gym's Taxi-v2
    - [LabTaxi](./assignments/P1-Intro/L8-Lab-Taxi/lab-taxi.ipynb)
 - Lesson 9: RL in Continuous Spaces
    - [Discretization](./assignments/P1-Intro/L9-RL-in-Continuous-Spaces/Discretization/Discretization.ipynb)
    - [Tile Coding](./assignments/P1-Intro/L9-RL-in-Continuous-Spaces/Tile-Coding/Tile_Coding.ipynb)

#### Part 2: Value-Based Methods
 - Lesson 2: Deep Q-Networks
    - [Deep Q-Learning Algorithm](./assignments/P2-Value-Based-Methods/L2-Deep-Q-Networks/exercise/Deep_Q_Network.ipynb)
    - ![Deep Q-Learning Algorithm GIF1](./assignments/P2-Value-Based-Methods/L2-Deep-Q-Networks/exercise/gifs/001.gif)

#### Part 3: Policy-Based Methods
 - Lesson 2: Introduction
    - [Cross Entropy](./assignments/P3-Policy-Based-Methods/L2-Intro-to-Policy-Based-Methods/cross-entropy/CEM.ipynb)
    - ![Cross Entropy GIF1](./assignments/P3-Policy-Based-Methods/L2-Intro-to-Policy-Based-Methods/cross-entropy/gifs/001.gif)
    - [Hill Climbing](./assignments/P3-Policy-Based-Methods/L2-Intro-to-Policy-Based-Methods/hill-climbing/Hill_Climbing.ipynb)
    - ![Hill Climbing GIF1](./assignments/P3-Policy-Based-Methods/L2-Intro-to-Policy-Based-Methods/hill-climbing/gifs/001.gif)
 - Lesson 3: Policy Gradient Methods
    - [REINFORCE](./assignments/P3-Policy-Based-Methods/L3-Policy-Gradient-Methods/reinforce/REINFORCE.ipynb)
    - ![REINFORCE GIF1](./assignments/P3-Policy-Based-Methods/L3-Policy-Gradient-Methods/reinforce/gifs/001.gif)
 - Lesson 4: Proximal Policy Optimization
    - [Pong using REINFORCE](./assignments/P3-Policy-Based-Methods/L4-Proximal-Policy-Optimization/pong-REINFORCE.ipynb)
    - ![Pong using REINFORCE GIF1](./assignments/P3-Policy-Based-Methods/L4-Proximal-Policy-Optimization/gifs-REINFORCE/001.gif)
    - [Pong using PPO](./assignments/P3-Policy-Based-Methods/L4-Proximal-Policy-Optimization/pong-PPO.ipynb)
    - ![Pong using PPO GIF1](./assignments/P3-Policy-Based-Methods/L4-Proximal-Policy-Optimization/gifs-PPO/001.gif)

#### Part 4: Multi-Agent Reinforcement Learning
 - Lesson 3: AlphaZero
    - [Tic Tac Toe](./assignments/P4-Multi-Agent-Reinforcement-Learning/L3-Case-Study-AlphaZero/tic-tac-toe/alphazero-TicTacToe.ipynb)
    - ![tic-tac-toe-1](./assignments/P4-Multi-Agent-Reinforcement-Learning/L3-Case-Study-AlphaZero/tic-tac-toe/images/machine-wins-1.png)
    - ![tic-tac-toe-2](./assignments/P4-Multi-Agent-Reinforcement-Learning/L3-Case-Study-AlphaZero/tic-tac-toe/images/machine-wins-2.png)

### Projects
#### Project 1: Navigation
 - [Navigation](./p1_navigation/Navigation.ipynb)
 - ![Navigation GIF1](./p1_navigation/results/gif/01.gif)
 
#### Project 2: Continuous Control
 - [Continuous Control](./p2_continuous-control/Continuous_Control.ipynb)
 - ![Continuous Control GIF1](./p2_continuous-control/results/gif/01.gif)

#### Project 3: Collaboration and Competition
 - [Collaboration and Competition](./p3_collab-compet/Tennis.ipynb)
 - ![Collaboration and Competition GIF1](./p3_collab-compet/results/gif/01.gif)

## Setup

### Intel GPU

### Intel® Arc™ B390 GPU

Please follow 
[PyTorch 2.11 Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-11.html)
article to install Intel GPU driver and deep learning essentials.

Tested on the following hardware specification and software version.

__Hardware Specification__
 - CPU: Intel® Core™ Ultra X7 Processor 358H
 - CPU Cores: 16 (4 Performance-cores, 8 Efficient-cores, and 4 Low Power Efficient-cores)
 - CPU Threads: 16
 - Memory: 32 GiB
 - iGPU: Intel® Arc™ B390 GPU
 
__Software Version__
 - Ubuntu 26.04
 - Intel Graphics Compute Runtime 26.14.37833.4
 - Python 3.14.4
 - PyTorch 2.11.0+xpu
 - Gymnasium 1.2.2
 - Arcade Learning Environment 0.11.2
 - NumPy 2.4.3
 - Matplotlib 3.10.9
 - Pandas 3.0.2

### Intel® Arc™ B580 Graphics

Please follow 
[PyTorch 2.7 Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-7.html)
article to install Intel GPU driver and deep learning essentials.

Tested on the following hardware specification and software version.

__Hardware Specification__
 - CPU: Intel® Core™ Ultra 9 Processor 285K
 - CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
 - CPU Threads: 24
 - Memory: 32 GiB
 - GPU: Intel® Arc™ B580 Graphics 
 - GPU Memory: 12 GiB
 
__Software Version__
 - Ubuntu 25.04
 - Intel Graphics Compute Runtime 24.52.032224
 - Intel Deep Learning Essentials 2025.0.2-6
 - Python 3.13.3
 - PyTorch 2.7.0+xpu
 - Gymnasium 1.1.1
 - NumPy 2.1.2
 - Matplotlib 3.10.1
 - Pandas 2.2.3

### Intel® Arc™ A770 Graphics

Although we are going to use PyTorch 2.7 testing, please follow 
[PyTorch 2.6 Prerequisites for Intel GPUs](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html)
article to install Intel GPU driver and deep learning essentials.

Tested on the following hardware specification and software version.

__Hardware Specification__
 - CPU: Intel® Core™ Ultra 9 Processor 285K
 - CPU Cores: 24 (8 Performance-cores and 16 Efficient-cores)
 - CPU Threads: 24
 - Memory: 32 GiB
 - GPU: Intel® Arc™ A770 Graphics
 - GPU Memory: 16 GiB
 
__Software Version__
 - Ubuntu 24.04.2 LTS
 - Intel Graphics Compute Runtime [25.05.32567.17](https://github.com/intel/compute-runtime/releases/tag/25.05.32567.17)
 - Intel Deep Learning Essentials 2025.0.2-6
 - Python 3.12.3
 - PyTorch 2.7.0+xpu
 - Gymnasium 1.1.1
 - NumPy 2.1.2
 - Matplotlib 3.10.1
 - Pandas 2.2.3

### Install Requirements

Install required packages.
```
$ sudo apt install python3-dev build-essential libopencv-dev swig ffmpeg python3-tk
```

Create virtual environment.
```
$ python3 -m venv pytorch_arc_env
$ source pytorch_arc_env/bin/activate
$ python -m pip install --upgrade pip
```

Install PyTorch and other required packages.
```
$ pip install torch==2.11 torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
$ cd DRLND
$ pip install --resume-retries 3 --upgrade -r requirements.txt
```

### Test Run

Activate virtual environment and setup variables.
```
$ source pytorch_arc_env/bin/activate
```

Detect GPU.
```
$ python -c "import torch; print(torch.xpu.is_available())"
```
```
True
```

### Run

Activate virtual environment and setup variables.
```
$ source pytorch_arc_env/bin/activate
```

Run the notebooks.
```
$ cd DRLND
$ jupyter lab
```

## Monitoring Tools

### Ubuntu
 - [top](https://man7.org/linux/man-pages/man1/top.1.html): CPU utilization and memory utilization for CPU and iGPU.
 - [Intel GPU top](https://manpages.ubuntu.com/manpages/noble/man1/intel_gpu_top.1.html): Intel iGPU and dGPU utilization.
 - [Intel PCM](https://github.com/intel/pcm): Intel CPU and iGPU power consumption.
   To support the latest Intel CPU products, compile Intel PCM manually using the following steps.
   ```
   $ sudo apt install cmake
   $ git clone https://github.com/intel/pcm.git
   $ cd pcm/
   $ mkdir build
   $ cd build
   $ cmake ..
   $ cmake --build . --parallel --config Release
   $ cd bin
   $ sudo ./pcm -silent
   ```
 - [Intel XPU-SMI](https://intel.github.io/xpumanager/smi_install_guide.html): Intel dGPU power consumption and memory utilization.
 - [NVIDIA SMI](https://docs.nvidia.com/deploy/nvidia-smi/index.html): NVIDIA dGPU utilization and power consumption.

### Windows

 - [HWiNFO](https://www.hwinfo.com/): CPU, iGPU, and dGPU utilization and power consumption.
