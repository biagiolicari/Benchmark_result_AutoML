import pandas as pd
import numpy as np
import time
import sys
import multiprocessing

from .MetaCVI import Meta_CVI
from .MetaAlgorithm import Algorithm
from .cvi import Validation
from multiprocessing import Process
from sklearn import metrics
from deap import base, creator, tools, algorithms

from .HyperPartitions import *
from .GeneticMethods import *
from pathlib import Path

script_directory = Path(__file__).resolve().parent.parent
relative_path_to_data = "../datasets"
    # Set the path to reach the data folder
data_path = script_directory / relative_path_to_data

class CSmartML:
	def __init__(self, filename, population, time_budget, meta_cvi=False, algorithm=None, cvi=None, dataset=None, result="multi"):

		self.time = time_budget
		self.filename = filename
		self.resultPreference = result

		# Load benchmark dataset or use uploaded dataset
		if dataset is None:
			
			self.data = pd.read_csv(relative_path_to_data / f'{dataset}.csv').iloc[:, :-1]
		else:
			self.data = dataset

		# Set algorithms and cvi by prescribed or meta-learning
		if meta_cvi:
			self.algorithm = Algorithm(filename, "distance", dataset).search()
			# self.algorithm = "birch"
			print(f"label : Getting reccomended metrics")
			print(f"algo: {self.resolve_abbr()}")
			self.cvi = Meta_CVI(filename, "distance", dataset).search(self.algorithm)
			# self.cvi = [['i_index', 1], ['modified_hubert_t', 1], ['banfeld_raferty', -1]]
			print(f"label : Generating hyper-partitions")
			print(f"algo: {self.resolve_metrics()}")
		else:
			self.algorithm = algorithm
			self.cvi = cvi
			
		# self.algorithm = "birch"
		# publish({
		# 	"label": "Getting recommended metrics...",
		# 	"algorithm": self.resolve_abbr()}, "message")
		# self.cvi = [['i_index', 1], ['modified_hubert_t', 1], ['banfeld_raferty', -1]]
		# publish({
		# 	"label": "Generating hyper-partitions...",
		# 	"metric": self.resolve_metrics()}, "message")

		# Creator: Assign Fitness Function (eg. multi-objective)
		fitness_weights = (np.float(self.cvi[0][1]), np.float(self.cvi[1][1]), np.float(self.cvi[2][1]))
		creator.create("FitnessMulti", base.Fitness, weights=fitness_weights)
		creator.create("Individual", list, fitness=creator.FitnessMulti)

		# Generate Hyper-Partitions
		hp_model = HyperPartitions(self.algorithm, self.data)
		hp_set = hp_model.hyper_param_combination()
		# Generate models for hyper-partition and assign mx/cx
		# Assign individual toolbox

		partition_id = 0
		self.toolbox = dict()
		self.partitions = dict()
		self.models = dict()

		print("label: Setting search methods for partitions...")
		for partition in hp_set:
			p_id = "P{}".format(partition_id)
			mx = self.Mode(partition, self.data, self.algorithm)
			toolbox = base.Toolbox()
			toolbox.register("individual", tools.initRepeat, creator.Individual, mx.gen_population, n=1)
			toolbox.register("population", tools.initRepeat, list, toolbox.individual)
			toolbox.register("evaluate", self.fitness_function)
			toolbox.register("select", tools.selNSGA2)
			if len(partition) > 1:
				toolbox.register("mate", mx.crossover)
				toolbox.register("mutate", mx.mutate)

			self.toolbox[p_id] = toolbox
			self.partitions[p_id] = partition
			partition_id += 1

		global gPartition
		partition = {}
		for k, v in self.partitions.items():
			value = None
			if len(v) > 1:
				value = "_&_".join(v)
			else:
				value = v[0]

			partition[k] = value


		gPartition = partition
		print(gPartition)

		print(f"label : Setting hyper-partitions...")
		print(f"algo: {self.resolve_partitions()}")

		# EA parameter initialization
		self.pop_size = population

		# Main toolbox for final selection
		self.main_toolbox = base.Toolbox()
		self.main_toolbox.register("evaluate", self.fitness_function)
		self.main_toolbox.register("select", tools.selNSGA2)

	# Return recommended cluster validation metrics
	def get_selected_metrics(self):
		return self.cvi

	# Return recommended cluster validation metrics
	def get_selected_algorithm(self):
		return self.algorithm

	# Get full algorithm name
	def resolve_abbr(self):
		ABRR_ALGORITHMS = {
			"db": "DBSCAN",
			"ag": "Agglomerated Clustering",
			"optics": "OPTICS",
			"ap": "Affinity Propagation",
			"kmeans": "K-Means Clustering",
			"birch": "Birch Clustering",
			"spectral": "Spectral Clustering",
			"meanshift": "MeanShift Clustering"
		}

		return ABRR_ALGORITHMS[self.algorithm]

	# Get full algorithm name
	def resolve_metrics(self):
		cvi = []
		for metric in self.cvi:
			cvi.append(metric[0].upper().replace("_", "-"))

		return [" & ".join(cvi), self.cvi]

	def resolve_partitions(self):
		partitions = []
		for partition in self.partitions.values():
			part_formatted = "_&_".join(partition)
			partitions.append(part_formatted)

		return partitions

	# Main Hyper-parameter search function
	# Assign EA or Random Search for hyper-partitions
	def search(self):
		print(f'label: Initializing search, creating processes..')
		func_dict = dict()
		# Run parallel if more than 1 hyper-partition
		if len(self.partitions) > 1:
			for key, value in self.partitions.items():
				population = self.toolbox[key].population(n=self.pop_size)
				if len(value) > 1:
					func_dict['ea_custom-' + key] = [key, self.time, population, self.toolbox[key], 0.7, 0.3]
				else:
					func_dict['random_search-' + key] = [key, self.time, self.pop_size, population, self.toolbox[key]]

		results = self.parallel_search(func_dict)
		#results = self.sequential_search(func_dict)

		print("label: Getting final solutions...")

		pop = list()
		for key, values in results.items():
			index = 0
			for value in values:
				if index < 10:
					pop.append(value)
					index += 1

		if self.resultPreference == "multi":
			results = self.main_toolbox.select(pop, 10)
		else:
			results = self.main_toolbox.select(pop, 1)
		# results = pop

		print("label: Preparing charts...")
		# Include NSGA2 Select for results on Fitness Function
		return results, self.algorithm

	# Create separate process for hyper-partition search
	# Store all results in res
	def parallel_search(self, partitions):

		manager = multiprocessing.Manager()
		res = manager.dict()
		process = []

		for partition in partitions:
			func_name = partition.split("-")[0]
			# func = getattr(sys.modules[__name__], func_name)
			func = globals()[func_name]
			partitions[partition].append(res)
			p = Process(target=func, args=tuple(partitions[partition], ))
			p.start()
			process.append(p)

		for proc in process:
			proc.join()

		return res
	
	def sequential_search(self, partitions):
		res = dict()
		for partition in partitions:
			func_name = partition.split("-")[0]
			func = globals()[func_name]
			partitions[partition].append(res)
			func(*partitions[partition])
		
		return res

	# Evaluate individual fitness
	def fitness_function(self, individual):

		try:
			clustering = individual[0].fit(self.data)
			labels = clustering.labels_
			validate = Validation(np.asmatrix(self.data).astype(np.float), np.asarray(self.data), labels)
			metric_values = validate.run_list([self.cvi[0][0], self.cvi[1][0], self.cvi[2][0]])
			return metric_values[self.cvi[0][0]], metric_values[self.cvi[1][0]], metric_values[self.cvi[2][0]]
		except Exception as e:
			print(e)
			return 0, 0, 0

	# Evaluate individual fitness: pareto front & rank
	# def fitness_function_mastered(self, population):


	class Mode:
		""" Inner Class for genetic methods of inidividual hyper-partitions"""
		def __init__(self, partition, data, algorithm):
			self.model = GeneticMethods(partition, data, algorithm)

		def gen_population(self):
			return self.model.generate_pop()

		def crossover(self, ind1, ind2):
			return self.model.crossover(ind1, ind2)

		def mutate(self, individual):
			return self.model.mutate(individual)


# Random search for single hyper-parameter tuning
def random_search(pid, t, s, population, toolbox, res):

	start_time = time.time()

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in population if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	ngen = 0
	while time.time() - start_time <= t:
		if ngen > 0:
			print("RANDOM: [NGEN - {}]".format(ngen))

		offspring = toolbox.select(population, len(population))

		# Generate random new individuals
		offspring = toolbox.population(n=s) + offspring

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

		# Temp variables to update Frontend
		trial_log = list()
		label = gPartition[pid]
		print("RS-PID: ", pid)

		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

			# Tuned parameter value, fitness value, current generation
			try:
				parameter_value = eval(str(ind[0]) + "." + label)
			except:
				parameter_value = 0

			if fit != (0, 0, 0):
				trial_log.append([parameter_value, fit, ngen])

		print(trial_log)
		# Replace the current population by the offspring
		population[:] = offspring

		ngen += 1

	res.update({pid: population})


# Custom Evolutionary Algorithm: With Time Budget
def ea_custom(pid, t, population, toolbox, cxpb, mutpb, res):

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in population if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit

	# Begin the [TIMED] generational process
	start_time = time.time()
	ngen = 0

	# Select the next generation individuals
	while time.time() - start_time <= t:
		if ngen > 0:
			print("EA: [NGEN - {}]".format(ngen))
		ngen += 1
		# offspring = toolbox.select(population, len(population))

		# Vary the pool of individuals
		offspring = algorithms.varOr(population, toolbox, 10, cxpb, mutpb)  # 20 for mu or lambda
		# offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

		# Temp variables to update Frontend
		trial_log = list()
		label = gPartition[pid]
		print("EA-PID: ", pid)
		print("EA Label: ", label)

		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

			# Tuned parameter value, fitness value, current generation
			# print(eval(str(ind[0]) + "." + label))
			if fit != (0, 0, 0):
				trial_log.append([0, fit, ngen])

		print(trial_log)

		# Replace the current population by the offspring
		# population[:] = offspring
		population[:] = toolbox.select(population + offspring, 10)  # Mu plus lambda : ea

	res.update({pid: population})