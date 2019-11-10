[//]: # (Image References)

[score_plot]: ./images/score_plot.png "Score Plot"

# Implementation

### Environment
Tennis environment has two agents working together in collaboration to achieve highest score. It receives actions from both agents and returns states, rewards and done flags for the both agents as well. Each action consists of two continuous values; horizontal move toward or away from the net and vertical move or jumping. These action values are in between -1 and 1 therefore the activation function for actor's output layer is hyperbolic tangent. The states consists of 8 values observing the ball and each racket. So each agent will have slightly different values based on the racket it holds. This implementation also uses two different policy neural networks for each agent and trains them separately. To solve this environment, the average score which is taken from the maximum reward of the two agents has to be +0.5 or higher over the past 100 episodes.

### Actor-Critic Methods
As this problem is highly unstable and tends to fall into local maxima, this implementation uses actor-critic methods to mitigate the issues. Actor model has low bias and it also helps to exit local maxima with its high variance nature whereas critic model helps the models to learn faster with its low variance nature although it has high bias. 

Actor model input layer only focuses on the states of its corresponding agent. On the other hand, critic model input layer observes the states of both agents. The nodes are 24 and 48 respectively. Actor model has two fully-connected hidden layers with 512 and 256 nodes as well as 2 nodes for the output layer for the actions. Critic model has three fully-connected hidden layers with 516, 256 and 128 nodes. The 516 nodes comes from 512 nodes of the output of input layer joined together with 2 nodes each of the actions from two agents. And the output layer of critic model is one node. Both actor and critic models utilize batch normalization and Xavier uniform weight initialization.

### Agent
To reduce correlations, this implementation uses replay buffer shared by both agents. It records states and actions taken by both agents at every time step. This implementation also injects Ornstein-Uhlenbeck noise into the actions to encourage exploration around the mean value.

This implementation uses Deep Deterministic Policy Gradient (DDPG) algorithm to train multiple agents with decentralized-actor and centralized-critic method. DDPG features like continuous action-space is useful to significantly reduce the number of nodes required by the neural networks. Other DDPG features like soft update is useful to perform a distinguishable targeted learning direction. Two types of clippings are also implemented. Gradient clipping is useful to avoid vanishing and exploding gradients whereas action clipping is useful to keep the actions within -1 and 1 due to noise injection.

Decentralized-actor means each actor learns from its own agent states and actions whereas centralized-critic means each critic learns from all agents states and actions. There is implementation that uses one neural networks model for all actors but this implementation uses one neural networks for each of the actors and trained separately.

### Training
This implementation updates the models 5 times at the end of each episode. The models reach the maximum score of +0.52 in 100 episodes window within 7678 episodes. Each episode score and the average score in 100 episodes window are shown below.

![Score Plot][score-plot]

### Hyperparameters
This implementation uses one million buffer size with 128 batch size. It uses 0.0001 actor learning rate and higher 0.001 critic learning rate. The critics have to pickup faster because the both actors actions have the initial values close to 1 that make both actors directly run to the net and just stay there. Critics have to quickly tell the actors to try other moves. This implementation also uses 0.99 discount factor, 0.001 soft update target parameters, no L2 weight decay and 0.1 standard deviation for action noise. Again, 0.2 standard deviation is too high making the average score in 100 episodes window stack at +0.3.

### Future Work
I am thinking of continuing both centralized-actor and centralized-critic method so the actors can have some understanding among them on the other actors states and actions. I started with this method with a very low average score in 100 episodes window at around +0.01 even after 20,000 episodes.
