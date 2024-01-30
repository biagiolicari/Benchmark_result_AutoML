import random
import numpy as np
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
	AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
	OPTICS, Birch


class KMEANS:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'n_clusters': random.randint(2, int(self.data_len / 4)),
			'init': random.choice(['k-means++', 'random']),
			'algorithm': random.choice(['auto', 'full', 'elkan']),
			'n_init': random.choice(list(range(10, 25))),
			'max_iter': random.choice(list(range(100, 350)))
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		return KMeans(**partition)


class MEANSHIFT:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'bandwidth': random.uniform(0.1, 2.5),
			'max_iter': random.randint(1, int(self.data_len / 4))
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		return MeanShift(**partition)


class DB:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'eps': random.uniform(0.1, 2.0),
			'min_samples': random.randint(5, 20)
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		return DBSCAN(**partition)


class AP:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'damping': random.uniform(0.5, 1),
			'max_iter': random.choice(list(range(150, 215))),
			'convergence_iter': random.choice(list(range(10, 25))),
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		return AffinityPropagation(**partition)


class Spectral:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'n_clusters': random.randint(2, int(self.data_len / 4)),
			'n_init': random.choice(list(range(10, 25))),
			'affinity': random.choice(['nearest_neighbors', 'rbf']),
			'eigen_solver': random.choice(['arpack', 'lobpcg', 'amg']),
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		if "eigen_solver" in self.sel_params:
			partition["n_components"] = random.randint(1, 11)

		if "affinity" in self.sel_params and partition["affinity"] == 'nearest_neighbors':
			partition["n_neighbors"] = random.randint(2, 5)
		else:
			partition["gamma"] = random.uniform(0.5, 2.0)

		return SpectralClustering(**partition)


class AG:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'distance_threshold': random.randint(2, int(self.data_len / 4)),
			'n_clusters': random.randint(2, int(self.data_len / 4))
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		if "n_clusters" in self.sel_params:
			partition["distance_threshold"] = None
		else:
			partition["n_clusters"] = None

		return AgglomerativeClustering(**partition)


class OP:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'min_samples': random.uniform(0.01, 0.20),
			'xi': random.uniform(0.01, 1.00),
			'db': random.uniform(0.5, 2.0),
			'eps': random.uniform(0.01, 1.00)
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		if "xi" in self.sel_params:
			partition["cluster_method"] = "xi"
		elif "db" in self.sel_params:
			partition["cluster_method"] = "db"

		return OPTICS(**partition)


class BIRCH:

	def __init__(self, params, data):
		self.sel_params = params
		self.data_len = len(data)
		self.all_params = {
			'threshold': np.random.choice(np.arange(0.01, 0.5, 0.001)),
			'branching_factor': np.random.choice(range(2, int(self.data_len / 5))),
			'n_clusters': np.random.choice(range(2, int(self.data_len / 5)))
		}

	def config(self):
		partition = dict()
		for params in self.sel_params:
			partition[params] = self.all_params[params]

		return Birch(**partition)

