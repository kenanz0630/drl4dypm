from machin.frame.algorithms.ddpg_per import DDPGPer
from machin.frame.algorithms.utils import soft_update
import torch
import torch.nn as nn
import numpy as np
from .pydybm.time_series.rnn_gaussian_dybm import RNNGaussianDyBM
from .pydybm.base.sgd import RMSProp
from scipy.signal import savgol_filter
from scipy.optimize import minimize, Bounds, LinearConstraint



eps = 1e-8



class CRPAgent:
	"""
	Constantly rebalanced portfolio

	"""

	def __init__(self, action_dim):

		self.action = np.ones(action_dim)/action_dim


	def get_action(self):
		return self.action







class RLAgent:
	"""
	Base off-policy DDPG agent with prioritized replay

	"""
	def __init__(self,
				 state_dim,						# n_feat
				 action_dim, 					# 1+n_assets
				 k, 							# time series length
				 network_params,
				 actor_learning_rate,
				 critic_learning_rate,
				 ipm_params = None, 			# IPM params
				 bcm_params = None, 			# BCM params
				 gamma=0.99,					# discount factor
				 batch_size=128,				# mini batch size
				 replay_capacity=int(1e6), 		# size of experience history
				 l2_reg=1e-6,					# L2 regularization weight
				 noise_mode='normal',			# distribution of action noise
				 noise_param=(0,0.01)			# params of action noise
				 ):


		# base params
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.k = k
		self.network_params = network_params
		self.actor_learning_rate = actor_learning_rate
		self.critic_learning_rate = critic_learning_rate
		self.gamma = gamma
		self.batch_size = batch_size
		self.replay_capacity = replay_capacity
		self.l2_reg = l2_reg
		self.noise_mode = noise_mode
		self.noise_param = noise_param

		self.ipm_active = False
		self.bcm_active = False



		# build IPM
		self.ipm_params = ipm_params
		if self.ipm_params:
			self.ipm_active = True
			self._build_ipm()
		else:
			self.ipm_dim = 0


		# build BCM
		self.bcm_params = bcm_params
		if self.bcm_params:
			self.bcm_active = True
			self._build_bcm()



		# build base learning agent
		self._build_model()





	def _build_model(self):

		actor = self._build_actor()
		actor_target = self._build_actor()
		critic = self._build_critic()
		critic_target = self._build_critic()

		optimizer = lambda params, lr: torch.optim.Adam(params, lr=lr, weight_decay=self.l2_reg)
		criterion = torch.nn.MSELoss(reduction='sum')

		# DDPG with prioritized replay
		self.ddpg_per = DDPGPer(actor, actor_target,
							 critic, critic_target,
							 optimizer=optimizer,
							 criterion=criterion,
							 batch_size=self.batch_size,
							 actor_learning_rate=self.actor_learning_rate,
							 critic_learning_rate=self.critic_learning_rate,
							 discount=self.gamma,
							 replay_size=self.replay_capacity)




	def _build_actor(self):
		return Actor(self.state_dim, self.action_dim, self.ipm_dim, self.k, self.network_params['actor'])     



	def _build_critic(self):
		return Critic(self.state_dim, self.action_dim, self.ipm_dim, self.k, self.network_params['critic'])




	def _build_ipm(self):
		"""
		Build IPM module based on NDybM

		"""

		params = self.ipm_params

		self.ipm_dim = params['input_dim']
		self.ipm_learning_rate = params['learning_rate']

		self.ipm = RNNGaussianDyBM(self.ipm_dim, self.ipm_dim, params['rnn_dim'], 
									spectral_radius=params['spectral_radius'], sparsity=params['sparsity'],
									delay=params['delay'], decay_rates=params['decay_rates'],
								   SGD=RMSProp())
		self.ipm.set_learning_rate(self.ipm_learning_rate)


		self.ipm_base_input_noise = lambda x: np.random.normal(params['noise_mean'], params['noise_std'], size=x)
		self.ipm_savgol_filter = lambda x: savgol_filter(x, window_length=params['filter_window_length'],
															polyorder=params['filter_polyorder'])


		self.ipm_loss = []



	def _build_bcm(self):
		"""
		Init bounds and constraints of BCM optimization problem
		"""

		n = self.action_dim
		params = self.bcm_params

		self.last_action = np.append([1], np.zeros(n-1)) # init weight (all cash)
		self.cost_bps = params['cost_bps']
		self.bcm_update_rate = params['update_rate']

		self.bcm_bounds = Bounds(np.zeros(n*2), np.ones(n*2))

		self.bcm_constraints = np.block([
			[np.eye(n), -1*np.eye(n)],
			[np.eye(n), np.eye(n)],
			[np.ones(n), np.zeros(n)]
			])




	# ======================================================================
	# Base methods

	def get_action(self, state, ipm_predict=None):
		if self.ipm_active:
			return self.ddpg_per.act_with_noise(
				{'state': state, 'ipm': ipm_predict}, noise_param=self.noise_param, mode=self.noise_mode
			)
		else:
			return self.ddpg_per.act_with_noise(
				{'state': state}, noise_param=self.noise_param, mode=self.noise_mode
			)





	def store_transition(self, experience):
		self.ddpg_per.store_transition(experience)





	def update(self, return_loss=False):
		"""
		Update actor anc critic networks
		"""

		if self.bcm_active:
			act_loss, value_loss = self._update_with_bcm()
		else:
			act_loss, value_loss = self.ddpg_per.update()


		if return_loss:
			return act_loss, value_loss




	def load(self, model_dir):
		"""
		Load model from given dir
		"""

		self.ddpg_per.load(model_dir)



	def save(self, model_dir):
		self.ddpg_per.save(mode_dir)





	# ======================================================================
	# IPM methods

	def ipm_init(self):
		"""
		Restart IPM prediction
		call at the beginning of each episode
		"""

		self.ipm.init_state()
		self.ipm_loss = []



	def ipm_predict_and_learn(self, in_step, out_step=None):
		prediction = self.ipm.predict_next()
		if out_step is not None:
			self.ipm.learn_one_step(out_step)
			self.ipm_loss.append(np.sum(np.square(prediction-out_step))) 

		in_step += self._generate_ipm_input_noise(in_step.shape)
		self.ipm._update_state(in_step)

		return prediction



	def get_ipm_loss(self):
		"""
		Return IPM loss as RMSE
		"""

		return np.sqrt(np.mean(self.ipm_loss))




	def _generate_ipm_input_noise(self, size):
		base = self.ipm_base_input_noise(size)
		smth = self.ipm_savgol_filter(base)

		return smth




	# ======================================================================
	# BCM methods

	def get_bcm_action(self, prices, next_prices):
		"""
		Get BCM one-step greedy action
		by solving optimization problem

		max (u_t+1)^w_t - c sum_i |w'_i,t - w_i,t|

		"""
		# objective 
		u = next_prices/prices
		obj = lambda x: self._bcm_objective(x, u)

		# linear constraint
		temp = np.dot(u, self.last_action)	
		w_end = self.last_action * u / temp		# weight at the end of current period 
		left_bnd = np.concatenate([-1*np.ones(self.action_dim), w_end, [1]])
		right_bnd = np.concatenate([w_end, 2*np.ones(self.action_dim), [1]])

		lin_constr = LinearConstraint(self.bcm_constraints, left_bnd, right_bnd)


		# solve optimization problem
		z0 = np.abs(w_end - self.last_action)
		x0 = np.append(self.last_action, z0) # use last action as init solution
		res = minimize(obj, x0, method='trust-constr', constraints=[lin_constr], bounds=self.bcm_bounds)

		self.last_action = res.x[:self.action_dim]



		return self.last_action



	def _bcm_objective(self, x, u):
		return -np.dot(u, x[:self.action_dim]) + self.cost_bps*np.sum(x[self.action_dim:])





	def _update_with_bcm(self):
		"""
		Update with BCM

		mostly copy from machin.frame.algorithms.ddpg_per
		"""

		mod = self.ddpg_per
		concatenate_samples = True

		mod.actor.train()
		mod.critic.train()

		# sample batch via prioritized replay
		batch_size, (state, action, reward, next_state, bcm_action, terminal, others), index, is_weight = \
		mod.replay_buffer.sample_batch(mod.batch_size, concatenate_samples,
									   sample_attrs=['state','action','reward','next_state','bcm_action','terminal','*'])

		# update critic network
		# - generate y_i using target actor and target critic
		with torch.no_grad():
			next_action = mod.action_transform_function(
				mod._act(next_state, True), next_state, others
				)
			next_value = mod._criticize(next_state, next_action, True)
			next_value = next_value.view(batch_size, -1)
			y_i = mod.reward_function(
				reward, mod.discount, next_value, terminal, others
				)

		# - critic loss
		cur_value = mod._criticize(state, action)
		value_loss = mod.criterion(cur_value, y_i.to(cur_value.device))
		value_loss = value_loss * torch.from_numpy(is_weight).view([batch_size,1]).to(value_loss.device)
		value_loss = value_loss.mean()


		# - update critic
		mod.critic.zero_grad()
		value_loss.backward()
		nn.utils.clip_grad_norm_(
			mod.critic.parameters(), mod.grad_max
			)
		mod.critic_optim.step()


		# update actor network
		# - actor loss
		cur_action = mod.action_transform_function(
			mod._act(state), state, others
			)
		act_value = mod._criticize(state, cur_action)


		act_policy_loss = -act_value.mean()
		
		# - add BCM loss
		act_policy_loss += self.bcm_update_rate * self._bcm_loss(cur_action['action'], bcm_action)

		# - update actor
		mod.actor.zero_grad()
		act_policy_loss.backward()
		nn.utils.clip_grad_norm_(
			mod.actor.parameters(), mod.grad_max
			)
		mod.actor_optim.step()



		# update target networks
		soft_update(mod.actor_target, mod.actor, mod.update_rate)
		soft_update(mod.critic_target, mod.critic, mod.update_rate)


		mod.actor.eval()
		mod.critic.eval()


		self.ddpg_per = mod



		return -act_policy_loss.item(), value_loss.item()




	def _bcm_loss(self, action, bcm_action):
		"""
		Compute log loss due to BCM action and actor's action

		action: tensor(batch_size, action_dim)
		bcm_action: List(batch_size)

		"""

		bcm_action = torch.stack(bcm_action).view(-1, self.action_dim)
		loss = bcm_action*torch.log(action+eps) + (1-bcm_action)*torch.log(1-action+eps)

		return -loss.mean()







	
























class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, ipm_dim, k, params):
		super(Actor, self).__init__()

		params_lstm = params['lstm']
		self.lstm  = nn.LSTM(state_dim, 
							 params_lstm['hidden_dim'], 
							 params_lstm['num_layers'],
							 batch_first=True,
							 bidirectional=True)


		params_fc = params['fc']
		n = len(params['fc'])
		fc = []


		self.lstm_out_dim = params_lstm['hidden_dim']*2*k
		fc.append(nn.Linear(self.lstm_out_dim+ipm_dim, params_fc[0])) 
		for i in range(1,n):
			fc.append(nn.Linear(params_fc[i-1], params_fc[i]))
		fc.append(nn.Linear(params_fc[-1], action_dim))

		self.fc = nn.ModuleList(fc)
		

		params_dropout = params['dropout']
		self.dropout = nn.Dropout(params_dropout)
		self.leakyReLU = nn.LeakyReLU()
		self.softmax = nn.Softmax(dim=1)




	def forward(self, state, ipm=None):
		a = self.leakyReLU(self.lstm(state)[0].reshape(-1,self.lstm_out_dim)) # LSTM output, all steps
		# a = self.leakyReLU(self.lstm(state)[0][:,-1]) # LSTM output, last step
		if ipm is not None:
			a = torch.cat([a, ipm], dim=1) # concatenate LSTM outputs with price prediction

		n = len(self.fc)
		a = self.leakyReLU(self.fc[0](a))
		a = self.dropout(a) # dropout after first fully connected layer
		for i in range(1,n-1):
			a = self.leakyReLU(self.fc[i](a))

		a = self.softmax(self.fc[-1](a))

		return a









class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, ipm_dim, k, params):
		super(Critic, self).__init__()

		params_lstm = params['lstm']
		self.lstm  = nn.LSTM(state_dim, 
							 params_lstm['hidden_dim'], 
							 params_lstm['num_layers'],
							 batch_first=True,
							 bidirectional=True)


		params_fc = params['fc']
		n = len(params['fc'])
		fc = []


		self.lstm_out_dim = params_lstm['hidden_dim']*2*k
		fc.append(nn.Linear(self.lstm_out_dim+ipm_dim+action_dim, params_fc[0])) 
		for i in range(1,n):
			fc.append(nn.Linear(params_fc[i-1], params_fc[i]))
		fc.append(nn.Linear(params_fc[-1], 1))

		self.fc = nn.ModuleList(fc)
		

		params_dropout = params['dropout']
		self.dropout = nn.Dropout(params_dropout)
		self.leakyReLU = nn.LeakyReLU()




	def forward(self, state, action, ipm=None):
		a = self.leakyReLU(self.lstm(state)[0].reshape(-1, self.lstm_out_dim)) # LSTM output, all step
		# a = self.leakyReLU(self.lstm(state)[0][:,-1]) # LSTM output, last step
		if ipm is not None:
			a = torch.cat([a, ipm, action], dim=1) # concatenate with action
		else:
			a = torch.cat([a, action], dim=1) # concatenate with action

		n = len(self.fc)
		a = self.leakyReLU(self.fc[0](a))
		a = self.dropout(a) # dropout after first fully connected layer
		for i in range(1,n-1):
			a = self.leakyReLU(self.fc[i](a))

		a = self.fc[-1](a)

		return a








