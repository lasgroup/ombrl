# OMBRL: Optimistic Exploration in Model-based RL
This repository provides an open-source implementation of algorithms for optimistic exploration in model-based reinforcement learning. It is based on the [MaxInfoRL](https://github.com/sukhijab/maxinforl) repository, which itself builds on [jaxrl](https://github.com/ikostrikov/jaxrl).

Among others, the repository includes implementations of the following algorithms:

- [SOMBRL: Scalable and Optimistic Model-Based RL](https://arxiv.org/abs/2511.20066) (Sukhija et al., NeurIPS 2025)
- COMBRL from our paper [Sample-efficient and Scalable Exploration in Continuous-Time RL](https://arxiv.org/abs/2510.24482) (Iten et al., ICLR 2026)
- SoftAE from our paper [Learning Soft Robotic Dynamics with Active Exploration](https://arxiv.org/abs/2510.27428) (Zheng et al., preprint 2025)
- R-OMBRL and SW-OMBRL from our preprint [Model-Based Reinforcement Learning for Control under Time-Varying Dynamics](https://arxiv.org/abs/2604.02260) (Iten et al., preprint 2026)

## Did this help you?

If you have any questions, feel free to reach out to us at `kiten[at]ethz[dot]ch`.

If you found this repository useful in your work, we would appreciate a citation:

```bibtex
@misc{sukhija2025sombrl,
  title         = {SOMBRL: Scalable and Optimistic Model-Based RL},
  author        = {Bhavya Sukhija and Lenart Treven and Carmelo Sferrazza and Florian Dörfler and Pieter Abbeel and Andreas Krause},
  year          = {2025},
  eprint        = {2511.20066},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2511.20066},
}

@misc{iten2026sampleefficient,
  title         = {Sample-efficient and Scalable Exploration in Continuous-Time RL},
  author        = {Klemens Iten and Lenart Treven and Bhavya Sukhija and Florian Dörfler and Andreas Krause},
  year          = {2026},
  eprint        = {2510.24482},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.24482},
}
```

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

5. add ombrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/to/ombrl```.

6. Launch experiments with the launcher: 
    ```
    python path/to/ombrl/experiments/experiment_name/launcher.py
    ```


### Remote Deployment on [euler.ethz.ch]([https://scicomp.ethz.ch/wiki/Main_Page](https://docs.hpc.ethz.ch/))

1. Set up remote development from your computer to Euler in either [PyCharm](https://www.jetbrains.com/help/pycharm/creating-a-remote-server-configuration.html#mapping) or [VSCode](https://code.visualstudio.com/docs/remote/ssh-tutorial).

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

5. add ombrl to your python path: ```PYTHONPATH=$PYTHONPATH:/path/on/euler/to/ombrl```. You can also add this to your ```.ombrl_setup``` file.

6. Launch experiments with the launcher: 
    ```
    python path/on/euler/to/ombrl/experiments/experiment_name/launcher.py
    ```
