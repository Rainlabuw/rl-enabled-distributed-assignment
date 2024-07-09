## Implementation of RL-Enabled Distributed Assignment (REDA)
This repository contains a streamlined implementation of RL-Enabled Distributed Assignment (REDA), the algorithm described to solve sequential assignment problems in 'Multi-Agent Reinforcement Learning for Sequential Satellite Assignment Problems'.

The code in this repository was based on the [EPyMARL framework](https://github.com/uoe-agents/epymarl), but has been modified to better fit REDA and the associated experiments.

## Installation
The Python environment parameters are listed in requirements.txt

**This works best with Python 3.8.12**, ideally installed in a conda environment.

Thus, the recommended steps are:
 - `conda create -n "myenv" python=3.8.12 ipython`
 - `conda activate myenv`
 - `pip3 install -r requirements.txt`
 - (Maybe necessary to get shapely to work) `conda install geos`

## Quick Guide to the Code
The critical REDA files are located in the following placaes:
 - `src/action_selectors/action_selectors.py` contains the code which builds the benefit matrix (\mathbf{Q}) from the neural network output and uses \alpha(\mathbf{Q} + \zeta) to make assignments.
 - `src/learners/sap_q_learner.py` contains the REDA learning loop, in which targets are calculated using rewards from the replay buffer and next-step rewards yielded by \alpha(Q). (Compare this to `src/learners/q_learner.py` for a clear view of the special implementation for REDA.)

## Running experiments from the paper
Several experiments from the paper have been set up in `src/experiments.py` for easy replication. Simply run:

`python3 src/experiments.py`

from the base directory with the appropriate tests selected to replicate these experiments.

To run the preset experiment training REDA, IQL, COMA, and IPPO from scratch in the dictator environment, uncomment `dictator_env_training()` in the `__main__` function of `src/experiments.py`.

To test the performance of pretrained REDA, IQL, and IPPO models as used in the paper, uncomment `constellation_env_test()` in `__main__`.

Finally, to run individual training runs of algorithms, use commands of the following format:

`python3 src/main.py --config=<alg_str> --env-config=<env_str> with <config_name>=<config_val>`

For example, to train REDA on the dictator environment with wandb tracking on, run:

`python3 src/main.py --config=dictator_reda --env-config=dictator_env with use_wandb=True`

To train IQL on the full constellation environment, using 10 parallel environments, run:

`python3 src/main.py --config=filtered_iql --env-config=constellation_env with runner=parallel batch_size_run=10`