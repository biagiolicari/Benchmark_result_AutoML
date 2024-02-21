from pathlib import Path
import pandas as pd
from src.MetaLearning import LearningPhase
from sklearn.preprocessing import StandardScaler

from src.ClusteringCS import ClusteringCS

from src.Experiments import DataGeneration

from src.MetaLearning import LearningPhase
from src.ClusterValidityIndices.CVIHandler import CVICollection
from src.Optimizer.OptimizerSMAC import SMACOptimizer
from neptune.utils import stringify_unsupported
from neptune import management
import neptune
import numpy as np
from sklearn import metrics
import csv
import subprocess

from MetaLearning.ApplicationPhase import ApplicationPhase
from MetaLearning import MetaFeatureExtractor
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
import numpy as np
import time

np.random.seed(9)

from pathlib import Path
import os

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
# Define the relative path to MetaKnowledgeRepository from the script directory
relative_path_mkr = "src/MetaKnowledgeRepository/"
# Set the mkr_path using the relative path
mkr_path = script_directory / relative_path_mkr
mf_set = MetaFeatureExtractor.meta_feature_sets[5]

relative_path_rw = "src/Experiments/RelatedWork/related_work"
related_work_path = script_directory / relative_path_rw


def prepare_dataset(relative_path:Path):
    dataset_dir = [f for f in relative_path.iterdir()]
    datasets_to_use = []
    true_labels_to_use = []
    dataset_names_to_use = []
    for idx, f in enumerate(dataset_dir):
        filename = f.name.split('.csv')[0]
        data = pd.read_csv(f'{f}')
        datasets_to_use.append(data.iloc[:, :-1].values.tolist())
        true_labels_to_use.append(data.iloc[:, -1].values.tolist())
        dataset_names_to_use.append(filename)
    return datasets_to_use, true_labels_to_use, dataset_names_to_use


def run_ml2dac_experiments():
    n_executions = 5
    n_warmstarts = 10 # Number of warmstart configurations (has to be smaller than n_loops)
    n_loops = 100 # Number of optimizer loops. This is n_loops = n_warmstarts + x
    limit_cs = True # Reduces the search space to suitable algorithms, dependening on warmstart configurations
    time_limit = 18000 # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
    cvi = "predict" #predict a cvi based on our meta-knowledge

    # Define the relative path to the data folder
    relative_path_to_data = "../datasets"
    # Set the path to reach the data folder
    data_path = script_directory / relative_path_to_data
    datasets_to_use, true_labels_to_use, dataset_names_to_use = prepare_dataset(data_path)

    col_names_df = ['dataset','framework','dbs','sil','ari','running_time_min', 'optimal_cfg']
    # File path
    file_path = 'ml2dac_experiments.csv'

    # Check if the file exists
    if not os.path.exists(file_path):
    # Create a DataFrame with the specified columns
        initial_df = pd.DataFrame(columns=col_names_df)

    # Write the DataFrame to a CSV file
        initial_df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created with columns: {col_names_df}")
    else:
        print(f"CSV file '{file_path}' already exists.")

    for i in range(n_executions + 1):
        for dataset, dataset_name, dataset_labels, in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
                result = pd.DataFrame(columns=col_names_df)  
                run = dict()
                run['dataset'] = dataset_name
                run['framework'] = 'ML2DAC'

                t0 = time.time()

                ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set)
                optimizer_result, additional_info = ML2DAC.optimize_with_meta_learning(dataset,
                                                                                        n_optimizer_loops=n_loops, 
                                                                                        cvi=cvi, time_limit=time_limit,
                                                                                        limit_cs=limit_cs,
                                                                                        dataset_name=dataset_name,
                                                                                        n_warmstarts=n_warmstarts)

                    # get the end time
                    # get the execution time
                running_time = round(time.time() - t0, 2)
                  
                online_opt_result_df = optimizer_result.get_runhistory_df()
                incumbent_stats = optimizer_result.get_incumbent_stats()
                incumbent = optimizer_result.get_incumbent()
                run['optimal_cfg'] = dict(incumbent)
                run['sil'] = metrics.silhouette_score(
                                             dataset, incumbent_stats["labels"])
                run['dbs'] = metrics.davies_bouldin_score(
                        dataset, incumbent_stats["labels"])
                run['ari'] = metrics.adjusted_rand_score(
                        dataset_labels, incumbent_stats["labels"])
                run['running_time_min'] = (round((running_time / 60), 3)) 
                result = result.append(run, ignore_index=True)
                result.to_csv('ml2dac_experiments.csv', mode='a', index=False, header=False)
                subprocess.run(['rm', '-rf', './smac/BO/'], check=True)
    


if __name__ == '__main__':
    run_ml2dac_experiments()
