# Deep reinforcement learning for dynamic portfolio management

### STAT 461 course project
#### Kenan Zhang

This repository is an implementation of the reinforcement learning model for dynamic portfolio management proposed in [Yu et al. (2019) Model-based Deep Reinforcement Learning for Dynamic Portfolio
Optimization](https://arxiv.org/abs/1901.08740). 

A presentation of this project is [here](https://youtu.be/6R_BrMVoSB0).


## Motivation

Dynamic portfolio management describes a process of sequentially allocate a collection of assets according to the stock prices 
in order to maixmize the long-term return. 
It natually falls into the famework of reinforcement learning, an agent learn an optimal policy through interacting with the environment. 
Hence, we may consider the portfolio reallocation as "action", the stock market as "environment", and the immediate investment return as "reward". 


###### RL system
<img src="/src/figs/rl_system.png" width="500">

###### Trading system
<img src="/src/figs/trading_system.png" width="500">

## Problem statement
Consider a portfolio of *m* assets and cash. We use vector **w** to denote the weight of each asset, and accordingly, the sum of weights equals 1.
Suppose the weight after the last reallocation is **w**<sub>t-1</sub>, at the end of the current time step, the weight shifts to **w**'<sub>t</sub> 
because of the stock price change. 
Then, we need to reallocate the portfolio such that the weights equal **w**<sub>t</sub>.


### MDP framework
Same as other reinforcement learning models, we need to first formulate the dynamic portfolio optimization problem as a Markov Decision Process (MDP). 
* State *s<sub>t</sub>*: A short history of normalized price. Consider the closing, low and high prices in the past *k* time steps. 
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
The actor and crtic networks have the same network structures but do not share parameters. Recall that state *s<sub>t</sub>* consists of a price matrix and the last action. The price matrix first passes through an LSTM RNN layer, which the authors called **Feature Extraction**. 

The output of LSTM layer is then concatenated with **w**<sub>t-1</sub> and passed to several fully connected layers, which the authors called **Feature Analysis**. 

The final output is then use to generate actions (for actor) and to estimate the state-action value (for critic). 

###### Network structure
<img src="/src/figs/network_arch.png" width="500">



### IPM
IPM is used to generate predictions about the next state, thus in this study, the input and output are the percentage change in the closing, low and high prices of each asset. 

IPM can be any time-seriese prediction model, but the authors proposed to use the [nonlinear dynamic Boltzman machine (NDyBM)](https://github.com/ibm-research-tokyo/dybm) because of its low time complexity for parameter update. 

In the learning process, after we observe the next state *s*<sub>t+1</sub>, IPM is called to predict the price change at the second next step **x**<sub>t+s</sub> and use the next state information to update its parameters. The current state *s*<sub>t</sub> and the next state *s*<sub>t+1</sub> are augmented with the current prediction **x**<sub>t+1</sub> and the new prediction **x**<sub>t+2</sub>, then stored to the replay buffer. 


### BCM
BCM is inspired by iminitation learning. It creates an expert who solves one-step greedy portfolio optimization problem at each step and use the actions of this expert to guide the learning of actor. The objective is defined as the log-rate return obtained at the next step less the contraction cost. 

In the learning process, after we observe the next state *s*<sub>t+1</sub>, BCM is called to solve the one-step greedy portfolio optimization problem. The optimal solution *ā<sub>t</sub>* is saved to the replay buffer along with other variables. 
When updating the actor, we compute an additional loss that measures the difference between the agent’s actions and the expert's actions, and use its gradient to update actor with a samll factor, along with the policy gradient derived by DDPG. 


## Implementation
Since there are no open-sourced codes, I implemented the model based on the paper. 

### Trading environment
The environment is built similar to the example given in Chp. 22 of book [Machine Learning for Algorithmic Trading](https://github.com/PacktPublishing/Machine-Learning-for-Algorithmic-Trading-Second-Edition). It has two main classes: 
* `DataSource`: load and processe the data, and provide state variables at each step 
* `TradingSimulator`: execute action and compute immediate reward 

`DataSource` and `TradingSimulator` are wrapped in `TradingEnvironment` to interact with the trading agent. The `take_step()` function takes action and current state as inputs, and return reward, next state and indicator of whether the trading period has ended. Each `state` is a tuple of three emelments: stock price, price matrix as input of actor/critic, and price percentage change as input of IPM. `TradingEnvironment` supports multiple agents. Hence, `actions` (`rewards`)	is a dictionary with agents' names as keys and agents' actions (rewards) as values.



    def take_step(self, actions, state):
        rewards = self.simulator.take_step(actions, state)
        next_state, end = self.data_source.take_step()
        
        return rewards, next_state, end

  

`DataSource`, `TradingSimulator` and `TradingEnvironment` are defined in [env.py](https://github.com/kenanz0630/drl4dypm/blob/master/src/drl4dypm/env.py).


### Trading agent

Four agents are implemented in current codes:
* Base agent: DDPG only
* IPM agent: DDPG + IPM
* BCM agent: DDPG + BCM
* Full agent: DDPG + IPM + BCM

For benchmark, another agent is developed that follows constantly rebalanced portfolio (CPR) strategy. 
It simply keeps equal weights for all asset over the trading period.  


All RL agents are instances of `RLAgent`. It is developed based on `DDPGPer` class in [machin](https://github.com/iffiX/machin/tree/af1b5d825e27a98deab7130eedbe1c2505dacf9d), a reinforcement library designed for pytorch. 

Main methods of `RLAgent` are:
* `get_action()`: call actor to generate a noisy or noise-free action
* `store_transition()`: store a transition to the replay buffer
* `update()`: update actor, critic and target networks

These methods take different inputs and perform different algorithms, according to the agent type. For instance, when IPM is implemented, `get_action()` takes both `state` and `ipm_predict` as inputs; when BCM is implemented, a self-defined update function is called 
where the additional expert auxiliary loss is considered.


The benchmark `CPRAgent` only has one method `get_action()` that returns the same weight vetor. 

Both `RLAgent` and `CPRAgent` are defined in [agent.py](https://github.com/kenanz0630/drl4dypm/blob/master/src/drl4dypm/agent.py).






## Experiments
The data used in this experiment are stock prices from the end of 2015 to early 2021. 
I selected six stocks to construct the portfolio, acrossing different industries:
* American Airline (AAL)
* CVS (CVS)
* FedEx (FDX)
* Ford (F)
* American International Group (AIG)
* Caterpillar (CAT)
The last three are also considered in the paper.


### Data processing

The raw data are minute-level stock price. Through data preprocessing, they are aggregated to daily closing, low and high prices and mapped to the same time horizon. The codes can be found in the notebook [data_processing](https://github.com/kenanz0630/drl4dypm/blob/master/src/data_processing.ipynb). 

###### Stock price over time
<img src="/src/figs/stock_price.png" width="800">


### Learning
An example model training and testing can be found in the notebook [main](https://github.com/kenanz0630/drl4dypm/blob/master/src/main.ipynb). In what follows, we will go through the main process step by step.

#### Set up parameters
We first set up parameters for the environment, agents, and the training process.

    # environment params
    trading_days = 252                                  # length of each trading period
    asset_names = ['AAL','CVS','FDX','F','AIG','CAT']   # names of selected stocks
    k = 10                                              # length of price history
    cost_bps = 1e-3                                     # rate of contraction cost
    path_to_data = 'data/stock_price_1D.csv'            # path to data

    # agent params
    num_assets = len(asset_names)                       # number of assets
    state_dim = 3*(1+num_assets)                        # dimension of price matrices 
    action_dim = 1+num_assets                           # dimension of action

    critic_learning_rate = 1e-3                         # learning rate of critic
    actor_learning_rate = critic_learning_rate * 0.5    # learning rate of actor


    # training params
    max_episode = 200                                   # number of training episodes
    min_episode_to_update = 10                          # minimum episode to start updating networks 
                                                        # (before that, there might not be enough samples) 



#### Create agents and environment
#### Training






