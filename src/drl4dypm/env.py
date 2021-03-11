import pandas as pd
import numpy as np
import copy


eps = 1e-6

class DataSource:
	"""
	Data source for trading environment

	1. Load, process and generate daily prices & volume data
	2. Provide data for each episode


	"""


	def __init__(self, num_steps, asset_names, k):
		self.num_steps = num_steps
		self.asset_names = asset_names
		self.num_assets = len(asset_names)
		self.k = k 							# time lag
		self.data = None
		self.feature = None
		self.offset = 0




	# def load_data(self, path_to_data,
	# 			  cols_rawdata=['adj_close', 'adj_low', 'adj_high'],
	# 			  cols_renamed=['close','low','high'],
	# 			  start_date='2000-01-01'):
	# 	self.step = 0
	# 	self.offset = None

	# 	print("Loading data from {} ...".format(path_to_data))
	# 	idx = pd.IndexSlice
	# 	with pd.HDFStore(path_to_data) as store:
	# 		df = (store['quandl/wiki/prices']
	# 			.loc[idx[start_date:,self.asset_names], cols_rawdata]
	# 			.dropna().sort_index())
		
	# 	df.columns = cols_renamed

	# 	self.data = df



	def load_data(self, path_to_data):
		self.step = 0
		self.offset = None
		self.data = pd.read_csv(path_to_data, index_col=['dt', 'ticker'], infer_datetime_format=True)





	def preprocess_data(self):
		"""
		Process data into three feature dict
		"""

		self.dict_state = {}
		idx = pd.IndexSlice

		for name in self.asset_names:
			self.dict_state[name] = self._compute_state_by_asset(self.data.loc[idx[:,name],:])


		self.data = self.data[(self.k-1)*self.num_assets:]




	def _compute_state_by_asset(self, data):
		"""
		Compute state by asset

		relative prices 		[p_t-k+1/p_t, ..., p_t-1/p_t, 1]
		normalized low price    [p^h_t-k+1/p_t, ..., p^h_t/p_t]
		normalized high price   [p^l_t-k+1/p_t, ..., p^l_t/p_t]
		relative price change   [(p_t-p_t-1)/p_t-1]

		P mtx: (data.length-(k-1), k)
		h: (data.length-(k-1), 3)

		"""

		dict_state = {}

		as_strided = np.lib.stride_tricks.as_strided 

		temp = data['close'].values
		mtx_close = as_strided(temp, (len(temp)-self.k+1, self.k), temp.strides*2)
		vect_close = ((temp[1:]-temp[:-1])/(temp[:-1]+eps)).reshape(-1,1)


		temp = data['low'].values
		mtx_low = as_strided(temp, (len(temp)-self.k+1, self.k), temp.strides*2)
		vect_low = ((temp[1:]-temp[:-1])/(temp[:-1]+eps)).reshape(-1,1)

		temp = data['high'].values
		mtx_high = as_strided(temp, (len(temp)-self.k+1, self.k), temp.strides*2)
		vect_high = ((temp[1:]-temp[:-1])/(temp[:-1]+eps)).reshape(-1,1)

		pt = mtx_close[:,-1].reshape(-1,1)+eps
		dict_state['P_close'] = mtx_close/pt
		dict_state['P_low'] = mtx_low/pt
		dict_state['P_high'] = mtx_high/pt

		n = len(vect_close) # data.length-1
		dict_state['h'] = np.concatenate([vect_close, vect_low, vect_high], axis=1)[self.k-2:]



		return dict_state



	def reset(self):
		"""
		Reset and start from a random day
		"""

		high = len(self.data)//self.num_assets - self.num_steps
		self.offset = np.random.randint(low=0, high=high)
		self.step = 0




	def take_step(self):
		"""
		Return data for current step

		data: (num_assets,)
		state: (k, 3*(1+num_assets))
		feat: (3*num_assets)
		"""

		data, state, feat = self._generate_state()

		self.step += 1
		end = self.step > self.num_steps

		return (data, state, feat), end



	def _generate_state(self):
		data = np.zeros(1+self.num_assets)
		data[0] = 1

		state = np.zeros((self.k, 3*(1+self.num_assets)))
		feat = np.zeros(3*self.num_assets)

		idx = pd.IndexSlice
		i = 0
		for name in self.asset_names:
			data[1+i] = self.data.loc[idx[:,name],'close'][self.offset + self.step]
			state[:,i+1] = self.dict_state[name]['P_close'][self.offset + self.step]
			state[:,i+2+self.num_assets] = self.dict_state[name]['P_low'][self.offset + self.step]
			state[:,i+3+2*self.num_assets] = self.dict_state[name]['P_high'][self.offset + self.step]

			feat[i] = self.dict_state[name]['h'][self.offset + self.step, 0]
			feat[i+self.num_assets] = self.dict_state[name]['h'][self.offset + self.step, 1]
			feat[i+2*self.num_assets] = self.dict_state[name]['h'][self.offset + self.step, 2]

			i += 1

		return data, state, feat









class TradingSimulator:
	"""
	Trading environment

	"""

	def __init__(self, num_assets, cost_bps, agent_names):
		self.num_assets = num_assets
		self.cost_bps = cost_bps
		self.agent_names = agent_names

		self.step = 0
		self.total_rewards = {}
		self.last_actions = {}
		self._reset_rewards_and_actions()
		
		self.last_prices = np.zeros(1+self.num_assets)








	def reset(self):
		self.step = 0
		self._reset_rewards_and_actions()
		self.last_prices.fill(0)
		



	def _reset_rewards_and_actions(self):
		for name in self.agent_names:
			self.total_rewards[name] = 0
			self.last_actions[name] = np.append([1], np.zeros(self.num_assets))





	def take_step(self, actions, prices):
		"""
		Compute reward based on each action, closing prices 
		and current portfolio weights

		"""

		
		rewards = {}

		if self.step == 0:
			u_t = np.ones(1+self.num_assets)								# relative price p(t)/p(t-1)
		else:
			u_t = prices/self.last_prices


		for name in self.agent_names:
			w_t_head = self.last_actions[name]

			temp = np.dot(u_t, w_t_head)	
			w_t_end = w_t_head * u_t / temp						 			# portfolio weight at the end of step
			c_t = self.cost_bps * np.linalg.norm(actions[name][1:]-w_t_end[1:], ord=1)		# cost factor
			
			rewards[name] = float(np.log(1-c_t) + np.log(temp))
			
			self.total_rewards[name] += rewards[name]
			self.last_actions[name] = actions[name]


		self.step += 1
		self.last_prices = prices

		return rewards



	def get_total_rewards(self):
		return self.total_rewards




class TradingEnvironment:
	"""
	Trading environment

	"""


	def __init__(self, num_steps, asset_names, k, cost_bps, agent_names, path_to_data):
		self.data_source = DataSource(num_steps, asset_names, k)
		self.data_source.load_data(path_to_data)
		self.data_source.preprocess_data()

		self.simulator = TradingSimulator(len(asset_names), cost_bps, agent_names)

		self.reset()



	def set_seed(self, seed=None):
		np.random.seed(seed)



	def reset(self):
		self.data_source.reset()
		self.simulator.reset()



	def init_step(self):
		return self.data_source.take_step()


	def take_step(self, actions, state):
		rewards = self.simulator.take_step(actions, state)
		next_state, end = self.data_source.take_step()

		return rewards, next_state, end



	def get_total_rewards(self):
		return self.simulator.get_total_rewards()


