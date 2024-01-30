import copy
import random

from .Algorithms import *


class GeneticMethods:

	def __init__(self, parameters, data, algorithm):
		# Differentiate mutable params from immutable
		self.params = parameters
		self.param_size = len(parameters)
		self.data = data
		self.algorithm = algorithm
		return

	def generate_pop(self):
		pop = self.custom_algorithm_config(self.params)
		return pop

	def mutate(self, pop):
		p = self.generate_pop()
		tpop = copy.deepcopy(pop)

		for i in range(self.param_size):
			pos = random.choice(list(range(0, self.param_size)))
			setattr(tpop[0], self.params[pos], getattr(p, self.params[pos]))

		return tpop,

	def crossover(self, pop, pop2):
		tpop, tpop2 = copy.deepcopy(pop), copy.deepcopy(pop2)

		for i in range(self.param_size):
			pos = random.choice(list(range(0, self.param_size)))
			setattr(tpop[0], self.params[pos], getattr(tpop2[0], self.params[pos]))
			setattr(tpop2[0], self.params[pos], getattr(tpop[0], self.params[pos]))

		return tpop, tpop2

	def custom_algorithm_config(self, params):

		algorithms = {
			'kmeans': KMEANS(params, self.data),
			'meanshift': MEANSHIFT(params, self.data),
			'db': DB(params, self.data),
			'ap': AP(params, self.data),
			'spectral': Spectral(params, self.data),
			'ag': AG(params, self.data),
			'optics': OP(params, self.data),
			'birch': BIRCH(params, self.data)
		}

		return algorithms[self.algorithm].config()

#
# import pandas as pd
# from sklearn.cluster import KMeans
#
# if __name__ == "__main__":
# 	file = "./Datasets/processed/{}.csv".format("iris")
# 	data = pd.read_csv(file, header=None, na_values='?')
# 	gm = GeneticMethods(["n_clusters", "init"], data, "kmeans")
# 	pop = KMeans(n_clusters=20, init="random")
# 	gm.mutate(pop)