# $Door(s)$: Junction State Estimation for Efficient Exploration in Reinforcement Learning

*Benjamin Fele, Jan Babič* <br>
*Jozef Stefan Institute, Jamova cesta 39, 1000 Ljubljana, Slovenia* <br>
*{benjamin.fele, jan.babic}@ijs.si*

[[paper]](https://proceedings.mlr.press)
[[teaser]](https://www.youtube.com)
[[demo]](https://www.youtube.com)

## Abstract

Exploration is one of the important bottlenecks for efficient learning in reinforcement learning, especially in the presence of sparse rewards. One way to traverse the environment faster is by passing through junctions, or metaphorical doors, in the state space. We propose a novel heuristic, $Door(s)$, focused on such narrow passages that serve as pathways to a large number of other states. Our approach works by estimating the state occupancy distribution and allows computation of its entropy, which forms the basis for our measure. Its computation is more sample-efficient compared to other similar methods and robustly works over longer horizons. Our results highlight the detection of dead-end states, show increased exploration efficiency, and demonstrate that $Door(s)$ encodes specific behaviors useful for downstream learning of various robotic manipulation tasks.

## How to run

### Environment setup

All needed python libraries are in the `environment.yml` file. You can directly replicate the environment with Anaconda. The code was tested with Python 3.11.

### Training

All the training commands used in the paper are available in file `train_all.py`, divided into the three training stages reported in the paper. Note the structure of these commands:
```
python train.py --config_path {path_to_config} {--additional_argument0 value0 ...}
```

Every command references a *configuration file* encoding general experimental hyperparameters. Please, inspect the files before running the code with your own problems. The configuation loading system takes care of additional referenced `.json` files in the configuration. Duplicate keys are overwritten following the order they are loaded at experiment initialization (duplicates are reported). Any additional arguments in the command itself are then treated as configuration parameters. For example, `--wrapper_kwargs.1.reward_threshold 10` will replace the `reward_threshold` initialization argument of the 1st environment wrapper.

### Additional code

Non-essential (and non-cleaned up) code pertaining convergence detection and preliminary grid-world experiments is available upon request.

## Acknowledgements

This work was supported by the Slovenian Research Agency (research core funding) under Grant P2-0076.
