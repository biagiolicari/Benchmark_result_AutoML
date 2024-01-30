"""
Implements the optimizers that are used for hyperparameter-tuning.
The implemented algorithms are the bayesian optimization, random search,
hyperband and a combination of bayesian optimization and hyperband (BOHB)
"""
import json
import logging
import random
import time
from math import inf

import pandas as pd
import numpy as np
from ConfigSpace.configuration_space import Configuration
from hpbandster.core.base_iteration import BaseIteration
from hpbandster.core.master import Master
from hpbandster.core.result import Result
from hpbandster.optimizers.config_generators.bohb import BOHB as BOHB_generator
from hpbandster.optimizers.config_generators.random_sampling import RandomSampling
import uuid
from abc import abstractmethod, ABC

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB, HyperBand
from scipy.optimize import OptimizeResult
from sklearn.datasets import make_blobs
from skopt import gp_minimize, dummy_minimize, BayesSearchCV, Optimizer
from skopt.space import Integer, Categorical
from smac.facade.func_facade import fmin_smac
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.sobol_design import SobolDesign
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import ExecuteTARun

from Algorithm import ClusteringAlgorithms
from Algorithm.ClusteringAlgorithms import run_kmeans, run_algorithm
from Metrics.MetricHandler import MetricCollection
from Utils import Helper
from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer, BOHBOptimizer, HyperBandOptimizer

OPT_ABBREVS = {
    # BayesOptimizer.get_name(): "BO",
    SMACOptimizer.get_name(): "BO",
    RandomOptimizer.get_name(): "RS",
    HyperBandOptimizer.get_name(): "HB",
    BOHBOptimizer.get_name(): "BOHB"}


def get_opt_by_abbrev(opt_abbrev):
    if opt_abbrev == OPT_ABBREVS[SMACOptimizer.get_name()]:
        return SMACOptimizer
    elif opt_abbrev == OPT_ABBREVS[RandomOptimizer.get_name()]:
        return RandomOptimizer
    elif opt_abbrev == OPT_ABBREVS[HyperBandOptimizer.get_name()]:
        return HyperBandOptimizer
    elif opt_abbrev == OPT_ABBREVS[BOHBOptimizer.get_name()]:
        return BOHBOptimizer


def get_best_hyperband_config(result):
    id2conf = result.get_id2config_mapping()
    inc_id = result.get_incumbent_id()
    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]

    # We have access to all information: the config, the loss observed during
    # optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']['k']
    return inc_config, inc_loss


def get_ta_runtimes_smac(out_dir):
    """
    parses the runtimes for the ta (algorithm + metric) from the runhistory.json file in the directory out_dir.
    :param out_dir:
    :return: tuple with first element the list that contains the runtime for the target algorithm in each iteration and
    the second element the list of runtimes for the optimizer since the start for each iteration
    """
    ta_times = []
    opt_times = []
    with open('{}/runhistory.json'.format(out_dir)) as json_file:
        data = json.load(json_file)
        data = data['data']
        for runs in data:
            run = runs[1]

            add_info = run[3]

            ta_time = add_info["metric_algo_time"]
            ta_times.append(ta_time)

            optimizer_time = add_info["opt_time"]
            opt_times.append(optimizer_time)

    return ta_times, opt_times


def get_algo_time_by_confid_and_budget(result, conf_id):
    runs_for_config = result.get_runs_by_id(conf_id)
    algo_time_by_conf_id_and_budget = {}
    for run in runs_for_config:
        algo_metric_time = run.info['algo_time'] + run.info['metric_time']
        algo_time_by_conf_id_and_budget[(run.config_id, run["budget"])] = algo_metric_time

    return algo_time_by_conf_id_and_budget


def get_runtime_for_iteration(conf_per_iteration, result):
    """
    Returns the runtime for each Hyperband iteration. Since multiple configs are executed in parallel we retrieve the
    runtime by taking the time_stamps
    :param conf_per_iteration:
    :param result:
    :return:
    """
    all_runs_for_iteration = []
    for conf in conf_per_iteration:
        runs = result.get_runs_by_id(conf)
        for run in runs:
            all_runs_for_iteration.append(run)

    start_of_optimization = result.get_all_runs()[0].time_stamps['started']
    end_of_iteration = max([run.time_stamps['finished'] for run in all_runs_for_iteration])
    return end_of_iteration - start_of_optimization


def get_overall_runtime(all_runs):
    starts = [run.time_stamps['started'] for run in all_runs]
    ends = [run.time_stamps['finished'] for run in all_runs]
    # Return the diff between the earliest started and the latest finished run
    return max(ends) - min(starts)


def parse_hyperband_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("Optimizer result is instance of Result, so either hyperband or bohb")

    learning_curve = result.get_learning_curves()
    mapping = result.get_id2config_mapping()

    for conf_id, i in zip(learning_curve, range(len(learning_curve))):
        # one config, but it can contain a list if that config is taken to next round and executed with higher budget
        budgets_for_config = learning_curve[conf_id][0]
        iteration = conf_id[0]

        # retrieve configs for each iteration -> we want to use them for retrieving the time for each iteration
        conf_per_iteration = list(filter(lambda conf: conf[0] == iteration, learning_curve))
        # algo_time_for_iteration = get_runtime_for_iteration(conf_per_iteration, result)

        # Workaround, we have to get the runs and the "info" field for the config
        # However, we cannot satisfy the correct ordering, so we save the algo time by conf_id and budget
        algo_time_by_conf_id_and_budget = get_algo_time_by_confid_and_budget(result, conf_id)
        iteration_runtime = get_runtime_for_iteration(conf_per_iteration, result)

        for budget_loss in budgets_for_config:
            score = budget_loss[1]
            budget = budget_loss[0]
            max_budget = max([x[0] for x in budgets_for_config])
            algo_time = algo_time_by_conf_id_and_budget[(conf_id, budget)]

            k = mapping[conf_id]['config']['k']
            if 'algorithm' in mapping[conf_id]['config']:
                algorithm = mapping[conf_id]['config']['algorithm']
            else:
                algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

            opt_result_dict[ResultsFields.budget_history].append(budget)
            # add one since it starts to count with 0
            # iterations.append(iteration+1)
            opt_result_dict[ResultsFields.iteration].append(iteration + 1)

            opt_result_dict[ResultsFields.score_history].append(score)
            opt_result_dict[ResultsFields.config_history].append(k)
            opt_result_dict[ResultsFields.algorithm_history].append(algorithm)
            opt_result_dict[ResultsFields.algorithm_metric_time_history].append(algo_time)

            logging.info(
                "config: {}, iteration: {}, budget: {}, loss: {}".format(mapping[conf_id]['config']['k'], iteration,
                                                                         budget_loss[0], budget_loss[1]))
            if max_budget == optimizer.max_budget and score < opt_result_dict.best_score:
                opt_result_dict.upate_best_config(score, k, algorithm)

        # We want to keep track of only the best config for the whole iteration, so we do add the best config
        # len(budgets_for_config) times, which is the number how often one config is evaluated
        opt_result_dict[ResultsFields.best_score_history].extend(
            [opt_result_dict.best_score for n in budgets_for_config])
        opt_result_dict[ResultsFields.best_config_history].extend(
            [opt_result_dict.best_k for n in budgets_for_config])
        opt_result_dict[ResultsFields.total_time_per_iteration].extend([iteration_runtime for n in budgets_for_config])
        opt_result_dict[ResultsFields.best_algorithm_history].extend(
            [opt_result_dict.best_algorithm for n in budgets_for_config])

    return opt_result_dict


def parse_random_opt_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("result is an OptimizeResult object (so either random or bo)")

    # parameters are list of parameters, but we only have one (k) so we are mapping it to the first
    # parameter
    parameter_values = list(map(lambda x: x[0], result.x_iters))
    if len(result.x_iters[0]) > 1:
        algo_history = list(map(lambda x: x[1], result.x_iters))
    else:
        algo_history = [ClusteringAlgorithms.KMEANS_ALGORITHM for x in result.x_iters]
    opt_result_dict[ResultsFields.config_history] = parameter_values
    opt_result_dict[ResultsFields.score_history] = result.func_vals
    opt_result_dict[ResultsFields.algorithm_history] = algo_history

    for ind, k in enumerate(parameter_values):
        algorithm = algo_history[ind]
        score = result.func_vals[ind]
        if score < opt_result_dict.best_score:
            opt_result_dict.upate_best_config(score, k, algorithm)

        opt_result_dict[ResultsFields.best_config_history].append(opt_result_dict.best_k)
        opt_result_dict[ResultsFields.best_score_history].append(opt_result_dict.best_score)
        opt_result_dict[ResultsFields.best_algorithm_history].append(opt_result_dict.best_algorithm)

    opt_result_dict[ResultsFields.algorithm_metric_time_history] = [algo_time + metric_time
                                                                    for algo_time, metric_time
                                                                    in zip(optimizer.algo_runtimes,
                                                                           optimizer.metric_runtimes)]
    opt_result_dict[ResultsFields.iteration] = [ind + 1 for ind in range(len(parameter_values))]
    opt_result_dict[ResultsFields.total_time_per_iteration] = optimizer.optimizer_runtimes
    return opt_result_dict


def parse_smac_result(result, optimizer):
    opt_result_dict = OptResultDict()
    logging.info("Optimizer result is SMBO (SMAC) result.")

    out_dir = result.output_dir
    # get the target algorithm runtimes for each evaluated configuration
    metric_algo_time_history, total_runtimes_iteration_from_start = get_ta_runtimes_smac(out_dir)
    opt_result_dict.set_field(ResultsFields.algorithm_metric_time_history, metric_algo_time_history)
    opt_result_dict.set_field(ResultsFields.total_time_per_iteration, total_runtimes_iteration_from_start)

    result = result.solver
    history = result.runhistory
    all_configs = history.get_all_configs()

    # save history for each config in each iteration
    for config in all_configs:
        score = history.get_cost(config)
        k = config['k']
        algorithm = config["algorithm"]

        if not algorithm:
            algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

        if score < opt_result_dict.best_score:
            opt_result_dict.upate_best_config(score, k, algorithm)

        # k_deviation_history.append(true_k - best_config)
        opt_result_dict.add_best_config_and_score()

    opt_result_dict.set_field(ResultsFields.iteration, [iteration for iteration, _ in enumerate(all_configs, 1)])
    opt_result_dict.set_field(ResultsFields.score_history, [history.get_cost(conf) for conf in all_configs])
    opt_result_dict.set_field(ResultsFields.config_history, [config['k'] for config in all_configs])
    opt_result_dict.set_field(ResultsFields.algorithm_history, [config['algorithm']
                                                                if config['algorithm']
                                                                else ClusteringAlgorithms.KMEANS_ALGORITHM
                                                                for config in all_configs])

    return opt_result_dict


def parse_result(result, optimizer):
    opt_result_dict = OptResultDict()

    # check if hyperband or bohb result
    if isinstance(result, Result):
        opt_result_dict = parse_hyperband_result(result, optimizer)

    # we don't have instance of Result so we should have either Random or SMAC optimizer
    elif isinstance(result, OptimizeResult):
        opt_result_dict = parse_random_opt_result(result, optimizer)

    # So should be SMAC
    elif isinstance(result, SMAC4HPO):
        opt_result_dict = parse_smac_result(result, optimizer)

    else:
        # Something weird happened
        logging.warning("Found unknown optimizer result {} !".format(result))

    return opt_result_dict


class ResultsFields:
    algorithm_history = "algorithm"
    iteration = "iteration"
    config_history = "k"
    budget_history = "budget"
    score_history = "score"
    best_config_history = "best config"
    best_score_history = "best score"
    # total optimizer time
    # total_optimizer_time = "total time"
    # total runtime for the optimizer per iteration since the start of the optimization
    total_time_per_iteration = "time per iteration"
    # k_deviation_history = "k_deviation"
    # runtime for algorithm and metric per iteration
    algorithm_metric_time_history = "algorithm + metric time"
    best_algorithm_history = "best algorithm"


class OptResultDict(dict):
    def __init__(self, *args, **kwargs):
        result_fields = ResultsFields()
        fields = dir(result_fields)
        [self.init_field(field) for field in fields if not field.startswith("__")]
        self.best_k = -1
        self.best_score = inf
        self.best_algorithm = ClusteringAlgorithms.KMEANS_ALGORITHM

    def init_field(self, field):
        self[getattr(ResultsFields, field)] = []

    def create_dataframe(self):
        if len(self[ResultsFields.budget_history]) == 0:
            self[ResultsFields.budget_history] = [-1 for n in self[ResultsFields.iteration]]
        return pd.DataFrame(self)

    def upate_best_config(self, score, k, algorithm):
        self.best_score = score
        self.best_k = k
        self.best_algorithm = algorithm

    def add(self, field, value):
        self[field].append(value)

    def set_field(self, field, values):
        self[field] = values

    def add_best_config_and_score(self):
        self[ResultsFields.best_config_history].append(self.best_k)
        self[ResultsFields.best_score_history].append(self.best_score)
        self[ResultsFields.best_algorithm_history].append(self.best_algorithm)

    def overwrite_by_dataframe(self, df):
        for key in df.keys():
            self[key] = df[key].values.tolist()
        return self

    def print(self):
        for k, v in self.items():
            print("values for key {k}: {v}".format(k=k, v=v))
