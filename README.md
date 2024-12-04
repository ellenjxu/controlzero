# AlphaZero for Continuous Control

Famous successes with [AlphaZero](https://arxiv.org/abs/1712.01815) have been in games such as Chess and Go, where there is a clean reward signal, and the state space is discrete. Yet these results don't transfer well to real world control environments, where there is a continuous action space and noise.

**Can AlphaZero-like approaches learn optimal low-level controls?**

This repo produces an AlphaZero agent for continuous controls in [~250 lines](https://github.com/ellenjxu/mcts-control/blob/main/train.py) of code.

## Results

Training takes ~6 min on my laptop running parallelized MCTS over 3M simulation steps.

<img src="https://github.com/user-attachments/assets/66961a05-b9bc-4502-b3a0-ffde95c19417" alt="image" width="500">

## How to use

To train AlphaZero Continuous, evaluate the learned policy, and generate plots:

```
python train.py
```

To run the online MCTS planner (useful for debugging search):

```
python run_mcts.py
```

## TODO
- increase lag
- keep track of state history, run on controls challenge
- work for general Gym environment
