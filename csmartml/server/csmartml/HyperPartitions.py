import random
from .GeneticMethods import GeneticMethods as GM
from sklearn.cluster import KMeans, MeanShift, DBSCAN, \
	AffinityPropagation, SpectralClustering, AgglomerativeClustering, \
	OPTICS, Birch


PARAMETERS = {
	'kmeans': {
		'prime': ['n_clusters'],
		'bench': ['n_init', 'max_iter']
	},
	'meanshift': {
		'prime': ['bandwidth'],
		'bench': ['max_iter']
	},
	'db': {
		'prime': ['eps', 'min_samples'],
		'bench': []
	},
	'ap': {
		'prime': ['damping'], #preference parameter applicable, but range?
		'bench': ['max_iter']
	},
	'spectral': {
		'prime': ['n_clusters'],
		'bench': ['n_init', 'eigen_solver', 'affinity']
	},
	'birch': {
		'prime': ['threshold', 'branching_factor', 'n_clusters'],
		'bench': ['n_clusters']
	},
	'ag': {
		'prime': ['distance_threshold', 'n_clusters'],
		'bench': []
	},
	'optics': {
		'prime': ['min_samples', 'xi', 'eps'],
		'bench': ['min_samples']
	},
}


class HyperPartitions:

	def __init__(self, algorithm, data):
		self.algorithm = algorithm
		self.data = data

	# Generate hyper-parameter combinations for hyper_partitions
	def hyper_param_combination(self):
		hyper_params = []

		# Selected algorithm
		algorithm = PARAMETERS[self.algorithm]

		# Loop through tunable parameters and form combinations
		for super_params in algorithm['prime']:
			hyper_params.append([super_params])
			if super_params not in algorithm['bench'] and len(algorithm['bench']) > 0:
				hyper_params.append([super_params] + algorithm['bench'])
			if len(algorithm['bench']) > 1:  # prob. unnecessary
				for sub_params in algorithm['bench']:
					if sub_params != super_params:  # prevent duplicate hyper-parameter combinations
						hyper_params.append([super_params, sub_params])

		print(hyper_params)
		return hyper_params

	# Generate hyper-partition models from for hyper_parameter combinations
	def generate_hyper_partition(self):
		partitions = []
		hp_combination = self.hyperparam_combination()

		for hp in hp_combination:
			gm = GM.GeneticMethods(hp, self.data, self.algorithm)
			if len(hp) == 1:
				partitions.append({'type': 'single', 'model': gm})
			else:
				partitions.append({'type': 'ga', 'model': gm})

		return partitions


