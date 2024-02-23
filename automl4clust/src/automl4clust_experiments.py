from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import metrics
import csv
from sklearn.datasets import make_blobs
import ConfigSpace as CS

from Algorithm import ClusteringAlgorithms
from Metrics.MetricHandler import MetricCollection
from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer
import ConfigSpace.hyperparameters as CSH

import time
import os

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
main_directory = script_directory.parent

N_LOOPS = 100
N_EXECUTIONS = 5
# Define the relative path to the data folder
RELATIVE_PATH_TO_DATASETS = "../datasets"

def prepare_dataset(path:Path):
    dataset_dir = [f for f in path.iterdir()]
    datasets_to_use = []
    true_labels_to_use = []
    dataset_names_to_use = []
    for idx, f in enumerate(dataset_dir):
        filename = f.name.split('.csv')[0]
        data = pd.read_csv(f'{f}')
        datasets_to_use.append(data.iloc[:, :-1].values.tolist())
        true_labels_to_use.append(data.iloc[:, -1].values.tolist())
        dataset_names_to_use.append(filename)
    return datasets_to_use, true_labels_to_use, dataset_names_to_use, 

def run_automl_four_clust():
    # Set the path to reach the data folder
    data_path = main_directory / RELATIVE_PATH_TO_DATASETS
    datasets_to_use, true_labels_to_use, dataset_names_to_use = prepare_dataset(data_path)
    col_names_df = ['dataset','framework','dbs','sil','ari','running_time_min', 'optimal_cfg']

    file_path="automl4clust_experiments.csv"
    # Check if the file exists
    if not os.path.exists(file_path):
    # Create a DataFrame with the specified columns
        initial_df = pd.DataFrame(columns=col_names_df)
    # Write the DataFrame to a CSV file
        initial_df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created with columns: {col_names_df}")
    else:
        print(f"CSV file '{file_path}' already exists.")

    for i in range(N_EXECUTIONS + 1):    
        for dataset, dataset_name, dataset_labels, in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
                result = pd.DataFrame(columns=col_names_df)  
                run = dict()
                run['dataset'] = dataset_name
                run['framework'] = 'AutoML4Clust'

                try:
                    t0 = time.time()
                    opt_instance = SMACOptimizer(dataset=dataset,
                                                metric=MetricCollection.SILHOUETTE, n_loops=N_LOOPS)

                    opt_instance.optimize()
                    running_time = round(time.time() - t0, 3)
                except:
                    continue
            
                opt_result = opt_instance.get_best_configuration()
                labels = ClusteringAlgorithms.run_algorithm(algorithm_name=opt_result['algorithm'], data_without_labels=dataset, k=opt_result['k']).labels

                run['optimal_cfg'] = dict(opt_instance.get_best_configuration())
                run['sil'] = metrics.silhouette_score(dataset, labels)
                run['dbs'] = metrics.davies_bouldin_score(dataset, labels)
                run['ari'] = metrics.adjusted_rand_score(dataset_labels, labels)   
                run['running_time_min'] = (round((running_time / 60), 3))
                result = result.append(run, ignore_index=True)
                result.to_csv('automl4clust_experiments.csv', mode='a', index=False, header=False)
    


if __name__ == '__main__':
    run_automl_four_clust()
