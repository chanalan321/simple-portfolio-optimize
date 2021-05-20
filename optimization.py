import numpy as np

class Optimization(object):
	"""docstring for Optimization"""
	def __init__(self, fixed_return, risk_free, returns, matrix):
		super(Optimization, self).__init__()
		self.fixed_return = fixed_return
		self.risk_free = risk_free
		self.returns = np.array(returns) - self.risk_free
		self.matrix = matrix
		self.weight = None

	def optimize(self):
		self.weight = np.dot(np.linalg.inv(self.matrix), self.returns)
		return self.weight

	def nonShortSales(self):
		temp = np.array(self.returns)
		new_m = np.c_[self.matrix * 2, temp.T, np.ones(len(self.returns))]
		temp = np.append(temp, [0, 0])
		new_m = np.r_[new_m, [temp]]
		temp = np.append(np.ones(len(self.returns)), [0, 0])
		new_m = np.r_[new_m, [temp]]

		b = np.zeros([len(self.returns), 1])
		b = np.append(b, [sum(self.returns), 1])
		self.matrix = new_m
		self.returns = b
		return self.optimize()