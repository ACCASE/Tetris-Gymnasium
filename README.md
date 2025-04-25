[![Python](https://img.shields.io/pypi/pyversions/gymnasium.svg)](https://badge.fury.io/py/tetris-gymnasium)
[![PyPI](https://badge.fury.io/py/gymnasium.svg)](https://badge.fury.io/py/tetris-gymnasium)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# A Fork of Tetris Gymnasium

My additions are contained within the folder ACCASE

# Required Packages

Pytorch: https://pytorch.org/get-started/locally/

```bash
pip install tetris-gymnasium
```

```bash
pip install gymnasium
```

```bash
pip install "gymnasium[other]"
```

```bash
pip install PyYAML
```

```bash
pip install opencv-python
```

```bash
pip install tyro 
```

```bash
pip install stable-baselines3
```

```bash
pip install tensorboard
```

```bash
pip install wandb
```

May require Microsoft C++ build tools: https://visualstudio.microsoft.com/downloads/?q=build+tools

Additionally, CUDA could be installed if compatible with your system.

# To run

All versions can be launched from the debug section of VSCode or alternatively via the command line. 

In VSCode, all configurations will be available from a drop down and executable with the play button

If running from the command line, add the --train flag to initiate training mode otherwise, the previously trained
policy in the corresponding runs folder will be used for simulation.

Via command line:
-----------------------------------------
Random Agent
Simulation
```bash
python .\ACCASE\Grouped_Action_DQN\Fully_Random.py TetrisRandom
```
Training
```bash
python .\ACCASE\Grouped_Action_DQN\Fully_Random.py TetrisRandom --train
```
-----------------------------------------
Manually Weighted Feature Vector
Simulation
```bash
python .\ACCASE\Grouped_Action_Hand_Crafted\Manual_Cost_Function.py TetrisManualCost
```
-----------------------------------------
DQN with Replay Buffer
Simulation
```bash
python .\ACCASE\Grouped_Action_DQN\DQN_Replay.py TetrisReplay
```
Training
```bash
python .\ACCASE\Grouped_Action_DQN\DQN_Replay.py TetrisReplay --train
```
-----------------------------------------
Double DQN
Simulation
```bash
python .\ACCASE\Grouped_Action_DQN\Double_DQN_Replay.py TetrisDoubleDQN
```
Training
```bash
python .\ACCASE\Grouped_Action_DQN\Double_DQN_Replay.py TetrisDoubleDQN --train
```
-----------------------------------------
DQN without Replay Buffer
Simulation
```bash
python .\ACCASE\Grouped_Action_DQN\DQN_Basic.py Tetris1
```
Training
```bash
python .\ACCASE\Grouped_Action_DQN\DQN_Basic.py Tetris1 --train
```

#

Based on Tetris Gymnasium:
https://github.com/Max-We/Tetris-Gymnasium
