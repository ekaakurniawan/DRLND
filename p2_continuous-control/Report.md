[//]: # (Image References)

[scores]: ./images/scores.png "Scores"
[average-score]: ./images/average-score.png "Average Score"
[hp-batchNorm-sigma02-scores]: ./images/hp-batchNorm-sigma02-scores.png "batchNorm sigma02 Scores"
[hp-batchNorm-sigma02-average-score]: ./images/hp-batchNorm-sigma02-average-score.png "batchNorm sigma02 Average Score"
[hp-noBatchNorm-sigma01-scores]: ./images/hp-noBatchNorm-sigma01-scores.png "noBatchNorm sigma01 Scores"
[hp-noBatchNorm-sigma01-average-score]: ./images/hp-noBatchNorm-sigma01-average-score.png "noBatchNorm sigma01 Average Score"


# Implementation

### Environment
This implementation uses 20 agents and therefore considered to have 20 independent environments. The state of each environment can be observed from 33 variables and each environment receives 4 continuous actions. The purpose of having 20 agents instead of just 1 agent is to help replay buffer to reduce correlation further. This will also help *10 updates after 20 steps* method (described below) to quickly collect more data. Hence, the term for solved environment is to have an average score of +30 or higher within 100 episodes window over all 20 agents.

### Actor-Critic Methods
This problem is considered to be highly unstable and can easily fall into local minima therefore actor-critic methods are selected for this implementation. Actor model with low bias but high variance is used to avoid local minima whereas critic model with low variance and high bias is used to make the learning faster. Both models has 33 nodes for input layer, two hidden layers and an output layer. For the first hidden layer, actor model has 400 nodes whereas critic model has 400 state nodes plus 4 action nodes. For the second hidden layer, both of them has 300 nodes. Finally, the output layer for actor model has 4 nodes ranging between -1 to 1 therefore hyperbolic tangent is used for the activation function whereas actor model has only 1 node without activation function. Both models use batch normalization as well as Xavier uniform weight initialization.

### Agent
Replay buffer together with 20 agents are used to reduce correlations. Ornstein-Uhlenbeck noise is also introduced to the action to encourage exploration.

The agent learns using Deep Deterministic Policy Gradient (DDPG) algorithm. Some features of DDPG algorithm being used are continuous action-space and soft update. Unlike Deep Q Network (DQN) with discrete action-space, DDPG output layer has only 4 nodes instead of hundreds or even thousands of nodes by discretize the action-space. Soft update also helps to reduce correlations. The implementation is similar to two neural networks like DQN but instead of occasionally updating the target network, soft update frequently update the network with small fraction of the weights. During training, gradient clipping is also implemented into local critic model weights to avoid vanishing and exploding gradients.

### Training
The strategy for training used is 10 model updates after 20 steps. This is useful to collect more vary data into replay buffer before training. With 20 agents and 20 steps, the data collected are as much as 400 with 160 sampling size per update is somewhat reasonable.

The target average score of +30 or higher in 100 episodes window was achieved in 207 episodes. The scores and average score are shown below.

![Scores][scores]
![Average Score][average-score]

### Hyperparameters
The two most important things to achieve the target are setting Ornstein-Uhlenbeck noise sigma to 0.1 and implementing batch normalization. Without batch normalization and noise sigma of 0.2, the average score stacked at maximum of around +2 with many hyperparameters had been tried. These include implementing up to 3 hidden layers with 1024 nodes, learning strategy of one update every step, learning strategy of 20 updates every step, increasing replay buffer size from 1e5 to 1e6, increasing actor learning rate from 1e-4 to 1e-3, increasing critic learning rate from 1e-3 to 1e-2, increasing L2 weight decay from 0 to 0.0001.

Adding batch normalization while still using noise sigma of 0.2 improved average score to around +8. Following are the scores and average score.

![batchNorm sigma02 Scores](hp-batchNorm-sigma02-scores)
![batchNorm sigma02 Average Score](hp-batchNorm-sigma02-average-score)

Without batch normalization but changing noise sigma down to 0.1 improved average score to almost hitting the target but then it collapsed. Following are the scores and average score.

![noBatchNorm sigma01 Scores](hp-noBatchNorm-sigma01-scores)
![noBatchNorm sigma01 Average Score](hp-noBatchNorm-sigma01-average-score)

It seemed the wider standard deviation of the noise causes the action off target.

### Future Work
Current implementation takes very long (more than 200 episodes) to really pickup on the average score. Modifying learning hyperparameters like soft update (tau) or learning rate for both actor and critic might help.
