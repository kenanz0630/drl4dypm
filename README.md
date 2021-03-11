# Deep reinforcement learning for dynamic portfolio management

### STAT 461 course project
#### Kenan Zhang

This repository is an implementation of the reinforcement learning model for dynamic portfolio management proposed in [Yu et al. (2019) Model-based Deep Reinforcement Learning for Dynamic Portfolio
Optimization](https://arxiv.org/abs/1901.08740). 


## Motivation

Dynamic portfolio management describes a process of sequentially allocate a collection of assets according to the stock prices 
in order to maixmize the long-term return. 
It natually falls into the famework of reinforcement learning, an agent learn an optimal policy through interacting with the environment. 
Hence, we may consider the portfolio reallocation as "action", the stock market as "environment", and the immediate investment return as "reward". 



## Problem statement
Consider a portfolio of *m* assets and cash. We use vector **w** to denote the weight of each asset, and accordingly, the sum of weights equals 1.
Suppose the weight after the last reallocation is **w**<sub>t-1</sub>, at the end of the current time step, the weight shifts to **w**'<sub>t</sub> 
because of the stock price change. 
Then, we need to reallocate the portfolio such that the weights equal **w**<sub>t</sub>.


### MDP framework
Same as other reinforcement learning models, we need to first formulate the dynamic portfolio optimization problem as a Markov Decision Process (MDP). 
* State *s<sub>t</sub>*: A short history of normalized price. Consider the closing, low and high prices in the past k time steps. 
Normalize them with the closing price at the current time step. This gives us three price matrices of dimension *k&times;(m+1)*.
* Action *a<sub>t</sub>*: The portfolio weight after reallocation at each time step, i.e., **w**<sub>t</sub>. 
* Reward *r<sub>t</sub>*: The log rate return at each time step log(*&rho;<sub>t</sub>/&rho;<sub>t-1</sub>*), 
where *&rho;<sub>t</sub>* is the portfolio value at the end of time step *t*. The shift in portfolio value is attributed to both price change and 
the transaction cost of reallocation. 



### DDPG algorithm
The learning algorithm used in this project is [DDPG](https://arxiv.org/abs/1509.02971), a deterministic policy gradient method.
It applies the idea of actor-critic method---learn a critic by minimizing the Bellman-error and an actor by maximizing the estimated value function. 
To stablize the learning, the targets in critic learning are computed using two seperate networks (called target critic and actor networks). 
Parameters of these two networks are updated as exponential average of those of the critic and actor. 


## Model framework
DDPG is a model-free learning algorithm. To further improve the performance, the authors proposed three modules to support the learning. In this project, I implemented two of them: infused prediction module (IPM) and behavioral cloning module (BCM). 

### Network structures of actor and critic



