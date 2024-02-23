import os
from pathlib import Path
import time

import pandas as pd
from sklearn.decomposition import PCA
from csmartml import csmartml as csm
from collections import OrderedDict
from sklearn import metrics, manifold
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics
import csv
import numpy as np
import time
from timeout_decorator import timeout, TimeoutError
import os

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent
main_directory = script_directory.parent

POP = 10 # Number of warmstart configurations (has to be smaller than n_loops)
TIME_LIMIT = 18000 # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
cvi = "predict" #predict a cvi based on our meta-knowledge
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
        datasets_to_use.append(data.iloc[:, :-1].copy())
        true_labels_to_use.append(data.iloc[:, -1].copy())
        dataset_names_to_use.append(filename)
    return datasets_to_use, true_labels_to_use, dataset_names_to_use


@timeout(5 * 60 * 60)  # Timeout set to 5 hours (in seconds)
def execute_csmartml_run(dataset, dataset_name, time_budget, population):
    st = time.time()
    comb = csm.CSmartML(filename=dataset_name,
                                dataset=dataset,
                                population=population,
                                time_budget=time_budget,
                                meta_cvi=True,
                                result="single")

    res, algorithm = comb.search()
    et = time.time()
    elapsed_time = et - st
    return res, algorithm, elapsed_time

def run_csmartml_experiments():
    # Set the path to reach the data folder
    data_path = main_directory / RELATIVE_PATH_TO_DATASETS
    datasets_to_use, true_labels_to_use, dataset_names_to_use = prepare_dataset(data_path)
    
    col_names_df = ['dataset','framework','dbs','sil','ari','running_time_min', 'optimal_cfg']
    file_path="csmartml_experiments.csv"
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
                run['framework'] = 'cSmartML'
                try:
                    res, predictions, running_time = execute_csmartml_run(dataset, dataset_name, TIME_LIMIT, pop)
                except TimeoutError:
                    print(f"Timeout: The fitting and predicting process for {dataset_name} exceeded 5 hours.")
                    continue                
                
                pop = res[0]
                cluster = pop[0].fit(dataset)

                try:
                    run['sil'] = metrics.silhouette_score(
                                                dataset, cluster.labels_)
                    run['dbs'] = metrics.davies_bouldin_score(
                            dataset, cluster.labels_)
                    run['ari'] = metrics.adjusted_rand_score(
                            dataset_labels, cluster.labels_)
                except:
                    print('error on calulate metrics')
                    continue
                    
                run['running_time_min'] = (round((running_time / 60), 3)) 
                result = result.append(run, ignore_index=True)
                result.to_csv('csmartml_experiments.csv', mode='a', index=False, header=False)
    


if __name__ == '__main__':
    run_csmartml_experiments()
