# ppo.cpp
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) written in C++. 

Collect simulated experience with multi-threaded batch environments implemented in C++ and train policy + value function with PyTorch C++ ([LibTorch](https://pytorch.org/cppdocs/installing.html)).

## Train humanoid
Train humanoid locomotion behavior in ~10 minutes (reward $\approx$ 5000) using 20 threads with [Intel Core i9-14900K](https://www.intel.com/content/www/us/en/products/sku/236773/intel-core-i9-processor-14900k-36m-cache-up-to-6-00-ghz/specifications.html) CPU to collect simulated experience from [MuJoCo](https://mujoco.org/) (C) and [Nvidia RTX 4090]() to train a neural network policy and value function with [PyTorch](https://pytorch.org/) (LibTorch) on [Ubuntu 22.04.4 LTS](https://releases.ubuntu.com/jammy/).

<img src="assets/humanoid.gif" alt="drawing" />

From the `build/` directory, run:
```sh
./run --env humanoid --train --visualize --checkpoint humanoid --device {cpu|cuda|mps}
```

The saved policy can be visualized:
```sh
./run --env humanoid --load humanoid_{x}_{y} --visualize --device {cpu|cuda|mps}
```

Visualize pretrained policy (requires Apple ARM CPU):
```sh
./run --env humanoid --load pretrained/humanoid_apple_arm --visualize --device cpu
```
 
## Installation
`ppo.cpp` should work with Ubuntu and macOS.

Dependencies: [abseil](https://github.com/abseil/abseil-cpp), [libtorch](https://pytorch.org/get-started/locally/), [mujoco](https://github.com/google-deepmind/mujoco)

### Prerequisites
Operating system specific dependencies:

#### macOS
Install [Xcode](https://developer.apple.com/xcode/).

Install `ninja`:
```sh
brew install ninja
```

#### Ubuntu
```sh
sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build clang-12
```

### Clone ppo.cpp
```sh
git clone https://github.com/thowell/ppo.cpp
```

### LibTorch
LibTorch (ie PyTorch C++) should automatically be installed by CMake. Manual installation can be performed (perform steps 1 and 2 below to create a /build directory first):

#### macOS
Install LibTorch 2.3.1 [download](https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.1.zip) and extract to `ppo.cpp/build`.

If you encounter warnings for malicious software for Torch:
System Settings -> Security & Privacy -> Allow

You might also need to:

```sh
brew install libomp
```

```sh
install_name_tool -add_rpath /opt/homebrew/opt/libomp/lib PATH_TO/ppo.cpp/libtorch/lib/libtorch_cpu.dylib
```

#### Ubuntu
Install LibTorch CUDA 12.1 2.3.1 [download](https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip) and extract to `ppo.cpp/build`

### Build and Run
1. Change directory:
```sh
cd ppo.cpp
```

2. Create and change to build directory:
```sh
mkdir build
cd build
```

3. Configure:

#### macOS
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja
```

#### Ubuntu
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12
```

4. Build
```sh
cmake --build . --config=Release
```

### Build and Run ppo.cpp using VSCode
[VSCode](https://code.visualstudio.com/) and 2 of its
extensions ([CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools)
and [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools))
can simplify the build process.

1. Open the cloned directory `ppo.cpp`.
2. Configure the project with CMake (a pop-up should appear in VSCode)
3. Set compiler to `clang-12`.
4. Build and run the `ppo` target in "release" mode (VSCode defaults to
   "debug").

### Command-line interface
Setup:
- `--env`: `humanoid`
- `--train`: train policy and value function with PPO
- `--checkpoint`: filename in `checkpoint/` to save policy
- `--load`: provide string in `checkpoint/` 
directory to load policy from checkpoint
- `--visualize`: visualize policy 

Hardware settings:
- `--num_threads`: number of threads/workers for collecting simulation experience [default: 20]
- `--device`: learning device [default: cpu, cuda, mps]
- `--device_sim`: simulation device [default: cpu, cuda, mps]
- `--device_type`: data type for device [default: float]
- `--device_sim_type`: data type for device_sim [default: double]

PPO settings:
- `--num_envs`: number of parallel learning environments for collecting simulation experience
- `--num_steps`: number of environment steps for each environment used for learning
- `--minibatch_size`: size of minibatch
- `--learning_rate`: initial learning rate for policy and value function optimizer
- `--max_env_steps`: total number of environment steps to collect
- `--anneal_lr`: flag to anneal learning rate
- `--kl_threshold`: maximum KL divergence between old and new policies
- `--gamma`: discount factor for rewards
- `--gae_lambda`: factor for Generalized Advantage Estimation
- `--update_epochs`: number of iterations complete batch of experience is used to improve policy and value function
- `--norm_adv`: flag for normalizing advantages
- `--clip_coef`: value for PPO clip parameter
- `--clip_vloss`: flag for clipping value function loss
- `--ent_coef`: weight for entropy loss
- `--vf_coef`: weight for value function loss
- `--max_grad_norm`: maximum value for global L2-norm of parameter gradients
- `--optimizer_eps`: epsilson value for Adam optimizer
- `--normalize_observation`: normalize observations with running statistics
- `--normalize_reward`: normalize rewards with running statistics

Evaluation settings:
- `--num_eval_envs`: number of environments for evaluating policy performance
- `--max_eval_steps`: number of simulation steps (per environment) for evaluating policy performance
- `--num_iter_per_eval`: number of iterations per policy evaluation

## Notes
This repository was developed to:
- understand the [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) algorithm
- understand the details of [Gym environments](https://github.com/openai/gym), including autoresets, normalization, batch environments, etc
- understand the normal distribution neural network policy formulation for continuous control environments
- gain experience with [PyTorch C++ API](https://pytorch.org/cppdocs/)
- experiment with code generation tools that are useful for improving development times, including: [ChatGPT](https://pytorch.org/cppdocs/) and [Claude](https://claude.ai/)
- gain a better understanding of where performance bottlenecks exist for PPO
- gain a better understanding of how MuJoCo models can be modified to improve steps/time
- gain more experience using [CMake](https://cmake.org/)

MuJoCo models use resources from [IsaacGym environments](https://github.com/isaac-sim/IsaacGymEnvs), [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), [MJX Tutorial](https://github.com/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb), and [dm_control](https://github.com/google-deepmind/dm_control)

PPO implementation is based on [cleanrl: ppo_continuous_action.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_continuous_action.py).
