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

All RL agents are instances of `RLAgent`. It is developed based on `DDPGPer` class in [machin](https://github.com/iffiX/machin/tree/af1b5d825e27a98deab7130eedbe1c2505dacf9d), a reinforcement library designed for pytorch. 

Main methods of `RLAgent` are:
* `get_action()`: call actor to generate a noisy or noise-free action
* `store_transition()`: store a transition to the replay buffer
* `update()`: update actor, critic and target networks



#### Base agent
The base agent implements DDPG algorithm. It builds two actor network and two critic network according to given network parameters. 

An example of network parameters is

    network_params = {                              # parameters of actor and critic networks
        'actor': {
            'lstm': {
                'hidden_dim': 20,                   # one-layer LSTM with hidden state dimension 20
                'num_layers': 1
            },
            'fc': [256,128,64,32],                  # number of nodes in fully connected layers 
            'dropout': 0.5,                         # dropout fraction
        },
        'critic': {
            'lstm': {
                'hidden_dim': 20,
                'num_layers': 1
            },
            'fc': [256,128,64,32],
            'dropout': 0.5,
        }
    }

`network_params['actor']` and `network_params['critic']` are used to construct actor and critic. In this example, both actor and critic have an LSTM layer, followed by 0.5 dropout and four fully connected layers. 


#### IPM agent

Upon the base agent, IPM agent add a price prediction module and thus requires additional parameters. 

An example of IPM parameters is
    
    # IPM params
    ipm_params = {
        'input_dim': 3*num_assets,                  # input dimension 3xm
        'learning_rate': 0.1**3,                    # learning rate of IPM 
        'rnn_dim': 20,                              # dimension of hidden state in RNN layer
        'delay': 3,                                 # NDyBM parameters
        'decay_rates': [0.1,0.2,0.5,0.8],
        'spectral_radius': 0.95,
        'sparsity': 0.5,
        'noise_mean': 0.,                           # smoothed noise added to inputs (see Yu et al., 2019)
        'noise_std': 0.01,
        'filter_window_length': 5,
        'filter_polyorder': 3
    }

In the curren version, IPM is default to be an `RNNGaussianDyBM`. Before each episode, `ipm_init()` is called to initialize the state of IPM. At each step, `ipm_predict_and_learn()` is called to predict next price change and update IPM with the new target. 



#### BCM agent

Upon the base agent, BCM agent add a behavioral cloning module and it requires two more parameters.

An example of BCM parameters is

    bcm_params = {
        'cost_bps': cost_bps,                       # trasaction cost rate                   
        'update_rate': 0.1                          # weight of auxiliary expert loss
    }


At each step, `get_bcm_action()` is called to solve the one-step greedy portfolio optimization problem and return the optimal solution. When updating networks, `_update_with_bcm()` is called instead of regular update function of `DDPGPer`, which compute the auxiliary expert loss and the corresponding policy gradient.



#### Full agent

Initialized with both `ipm_params` and `bcm_params`.



#### CPR agent
For benchmark, `CPRAgent` is developed that follows constantly rebalanced portfolio (CPR) strategy. 
It simply keeps equal weights for all asset over the trading period.  



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
<img src="/src/figs/stock_price.png" width="600">


### Learning
An example model training and testing can be found in the notebook [main](https://github.com/kenanz0630/drl4dypm/blob/master/src/main.ipynb). 

In the example, we define one environment and five agents. Each time follows five steps:
1. Each agent generates action using current state
2. The environment takes these actions as inputs, evaluates the rewards and return the next state
3. Agents with IPM predict the next price change and update IPM model
4. Agents with BCM generate greedy actions
5. The current state, actions, rewards, the next state, along with IPM prediction and BCM action, are stored to replay buffer of each agent



### Evaluation
Following the paper, I used four metrics to evaluate the performance:
1. Annual return: mean of portfolio return *&mu*
2. Annual volatility: standard deviation of portfolio return *&sigma*
3. Sharpe ratio: return per unit risk *&mu/&sigma*
4. Sortino ratio: return per unit harmful volatility *&mu/&sigma<sub>d</sub>*




### Results
The figure below shows the total reward against training episodes. The BCM and full agents quickly achieve a pretty good learning performance while the reward of the other three remain close to zero. Specifically, BCM agent performs even better than the full agent. This also indicates that the behavior cloning module plays a major role in the full agent. 

#### Total reward against training episode
<img src="/src/figs/cmp_all.png" width="600">


The figure below shows how reward increases in a single episode. Again, BCM and full agents perform much better than the other three and they grow in parallel. 
On the other hand, the curves of base and IPM agents almost overlap each other and they perform slightly better than the benchmark.


#### Total reward in a single episode
<img src="/src/figs/test_sing_epi_r.png" width="600">


Follows figures illustrate the actions of each agent over the trading period. First, base and IPM agent act almost the same and they both assign all weight to CAT, which also has the most stable growth over time. On the other hand, BCM and full agents reallocate the portfolio more aggressively and frequently, while do not show preference on any single asset.


#### Actions in a single episode
Base agent            |  IPM agent
:-------------------------:|:-------------------------:
<img src="/src/figs/test_sing_epi_a_base.png" width="400">  |  <img src="/src/figs/test_sing_epi_a_ipm.png" width="400">


BCM agent            |  Full agent
:-------------------------:|:-------------------------:
<img src="/src/figs/test_sing_epi_a_bcm.png" width="400"> | <img src="/src/figs/test_sing_epi_a_full.png" width="400">



Finally, the table below reports the metrics computed over 100 test episodes. 
Again, base and IPM agents perform almost the same. The annual return is 4%, slightly better than the benchmark CRP agent with zero return.
BCM and full agents perform crazily well but also show large volatility. However, the difference between Sharpe ratio and Sortino ratio indicates that the large variation more often realizes above the mean value. 


Metric | Base | IPM | BCM | Full| CRP
-----|-----|-----|-----|-----|-----
Ann. return | 0.04 | 0.04 | 5.02 | 4.65 | -0.03
Ann. volatility | 0.11 | 0.11 | 5.06 | 4.90 | 0.07
Sharpe ratio | 0.38 | 0.38 | 0.99 | 0.95 | -0.38
Sortino ratio | 0.86 | 0.86 | 8.30 | 8.77 | -0.69


## Summary

In this project, I implemented a deep reinforcement learning model for dynamic portfolio management following [Yu et al. (2019)](https://arxiv.org/abs/1901.08740).  
I investigate how the proposed IPM and BCM could help improve the baseline model-free algorithm. Specifically, IPM is applied to enrich the state information and BCM enables the agent to imitate an expert.

The experiments show that all RL agents perform better than the benchmark trading strategy, but the ones with BCM works much better. However, the results may not be reliable because the data size is not very large. And, due to the lack of time, I only used the hyperparameters proposed in the paper. So I will continue the hyperparameter tuning in the future. 
To tackle the first issue, the paper also proposed a data augmentation module based on GAN. It can generate synthetic data for the training of RL agents. And this will also be the future work. 

