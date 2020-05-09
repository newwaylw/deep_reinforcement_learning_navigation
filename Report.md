### Summary

This project implements a 'deep' reinforcement agent that learns to navigate a 2D world and collect 'Bananas' as rewards.

The learning agent is represented as a two fully connected layer of neural network, with Leaky ReLU activation fuction in each layer. The second layer will have half the units as the first layer.

The agent follows the standard Q learning process, but uses a memory replay (size 10^6) and 2 identical neural networks (Double Q-Learning) to stablizes learning.

The agent follows a \(\epsilon\)-greedy policy, using a decay rate of 0.995 and capped at 0.01

### Performance

The environment is solved after 560 episodes (with an average score >13.0 over 100 episodes), the best score is achieved in episode 1671 with best score 16.05 

### Future work
Idears to improve the performance
1. Grid search on hyper-parameter optimisation
2. [Prioritized Replay](https://arxiv.org/pdf/1511.05952) to increase the sample rate of memories with high expected learning progress,
as measured by the magnitude of their temporal-difference (TD) error. 