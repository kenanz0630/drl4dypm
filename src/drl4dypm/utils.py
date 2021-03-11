import pandas as pd
import numpy as np

eps = 1e-6

def evaluation(values, window=100):
	df = pd.DataFrame(data=values)

	metrics = {
	'sharpe_ratio': _compute_sharpe_ratio(df, window),
	'sortino_ratio': _compute_sortino_ratio(df, window),
	'annual_volatility': _compute_annual_volatility(df, window),
	'annual_return': _compute_annual_return(df, window)
	}

	return metrics



def _compute_sharpe_ratio(df, window):
	mean = df.rolling(window).mean()
	std = df.rolling(window).std()

	return mean/std



def _compute_sortino_ratio(df, window):
	mean = df.rolling(window).mean()
	# fnc = lambda x: x+1
	negsid = df.rolling(window)
	negstd = df.rolling(window).apply(_compute_downside_deviation)

	return mean/negstd


def _compute_downside_deviation(x):
	mean = np.mean(x)
	idx_neg = (x-mean) < 0
	diff_neg = x[idx_neg] - mean

	return np.std(diff_neg)


def _compute_annual_volatility(df, window):
	return df.rolling(window).std()





def _compute_annual_return(df, window):
	return df.rolling(window).mean()





		




