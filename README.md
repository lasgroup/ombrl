# OMBRL: Optimistic Exploration in Model-based RL
This is an open-source implementation of algorithms for optimistic exploration in model-based RL settings, based on the [MaxInfoRL](https://github.com/sukhijab/maxinforl) repository, which in turn is based on [jaxrl](https://github.com/ikostrikov/jaxrl). Among others, you will find an implementation of the COMBRL algorithm from our paper [Sample-efficient and Scalable Exploration in Continuous-Time RL](https://arxiv.org/abs/2510.24482v1) and the SoftAE algorithm from our paper [Learning Soft Robotic Dynamics with Active Exploration](https://arxiv.org/abs/2510.27428). This repository is currently under construction, so be sure to check back soon - more info to come!

## Getting started

### Local Installation

1. Requirements:  
    - Python >=3.11
    - CUDA >= 12.1
    - cudnn >= 8.9

3. Install [JAX](https://jax.readthedocs.io/en/latest/installation.html) either on CPU or GPU:
    ```shell
    pip install -U "jax[cpu]"
    pip install -U "jax[cuda12]"
    ```

2. Install with a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):
    ```shell
    conda create -n ombrl python=3.11 -y
    conda activate ombrl
    git clone https://github.com/lasgroup/ombrl.git
    pip install -e .
    ```
    This also installs the [jaxrl](https://github.com/ikostrikov/jaxrl) and [MaxInfoRL](https://github.com/sukhijab/maxinforl) libraries.

4. set up [wandb](https://docs.wandb.ai/quickstart)

5. add ombrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/to/ombrl```. You can also add this to your .bashrc.

6. Launch experiments with the launcher: 
    ```
    python path/to/ombrl/experiments/experiment_name/launcher.py
    ```


### Remote Deployment on [euler.ethz.ch](https://scicomp.ethz.ch/wiki/Main_Page)

1. Set up remote development from your computer to Euler in either [PyCharm](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#mapping) ~~or [VSCode](https://code.visualstudio.com/docs/remote/ssh-tutorial)~~.

5. Set up git protocols on Euler: [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

2. Set up a .ombrl_setup file on your login node:
    ```shell
    export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
    export TF_FORCE_GPU_ALLOW_GROWTH=true
    export TF_DETERMINISTIC_OPS=0

    module load stack/2024-06
    module load gcc/12.2.0
    module load eth_proxy
    module load python/3.11.6

    PYTHONPATH=$PYTHONPATH:path/on/euler/to/ombrl
    export PYTHONPATH
    ```
    Source it with `source .ombrl_setup`.

3. Create a [miniconda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or a [python virtual environment](https://docs.python.org/3.11/library/venv.html#creating-virtual-environments).

5. activate virtual environment:
    ```shell
    source path/on/euler/to/venv/bin/activate
    ```

5. Install Jax for GPU (see [the JAX documentation](https://jax.readthedocs.io/en/latest/installation.html))
    ```shell
    pip install "jax[cuda12]"
    ```

4. git clone and pip install the ombrl library:
    ```shell
    git clone https://github.com/lasgroup/model-based-rl.git
    pip install .
    ```

5. set up [wandb](https://docs.wandb.ai/quickstart) on Euler

5. add ombrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/on/euler/to/ombrl```. You can also add this to your .bashrc or .mbrl_setup file.

6. Launch experiments with the launcher: 
    ```
    python path/on/euler/to/ombrl/experiments/experiment_name/launcher.py
    ```