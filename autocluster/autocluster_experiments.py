import autocluster
from autocluster import AutoCluster, get_evaluator, MetafeatureMapper
from sklearn import datasets
from collections import Counter
from sklearn.metrics.cluster import v_measure_score
import pandas as pd
from pathlib import Path
import neptune
from sklearn import metrics

# import necessary packages
import glob
import os
import sys
import logging
import json
import random
import traceback
import numpy as np
import pandas as pd
import autocluster
from autocluster import AutoCluster, PreprocessedDataset, get_evaluator, LogHelper, LogUtils, MetafeatureMapper, calculate_metafeatures

from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from datetime import datetime

import time
from timeout_decorator import timeout, TimeoutError

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent

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
def fit_and_predict(cluster, dataset, dataset_name, fit_params):
    st = time.time()
    res = cluster.fit(**fit_params)
    predictions = cluster.predict(dataset, plot=False,
                                  save_plot=False, file_path='{}/{}.png'.format(Path('images'), dataset_name))
    et = time.time()
    elapsed_time = et - st
    return res, predictions, elapsed_time

def run_autocluster():
    n_loops = 100
    
    # Define the relative path to the data folder
    relative_path_to_data = "../datasets"
    # Set the path to reach the data folder
    data_path = script_directory / relative_path_to_data
    datasets_to_use, true_labels_to_use, dataset_names_to_use = prepare_dataset(data_path)

    col_names_df = ['dataset','framework','dbs','sil','ari','running_time_min', 'optimal_cfg']
    seed = [27, 1234]

    file_path="autocluster_experiments.csv"
    # Check if the file exists
    if not os.path.exists(file_path):
    # Create a DataFrame with the specified columns
        initial_df = pd.DataFrame(columns=col_names_df)
    # Write the DataFrame to a CSV file
        initial_df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created with columns: {col_names_df}")
    else:
        print(f"CSV file '{file_path}' already exists.")
    
    for s in seed:
        for dataset, dataset_name, true_labels in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
                print(dataset_name)     
                result = pd.DataFrame(columns=col_names_df)
                run = dict()
                run['dataset'] = dataset_name
                run['framework'] = 'autocluster'

                general_metafeatures = MetafeatureMapper.getGeneralMetafeatures()
                numeric_metafeatures = MetafeatureMapper.getNumericMetafeatures()
                benchmark_metafeatures_table_path = Path(
                    "experiments/metaknowledge/benchmark_silhouette_metafeatures_table.csv")
                
                metafeatures_table = pd.read_csv(benchmark_metafeatures_table_path, sep=',', header='infer')


                fit_params = {
                "df": dataset,
                "cluster_alg_ls": ['KMeans', 'GaussianMixture', 'Birch', 'MiniBatchKMeans', 'AgglomerativeClustering', 'OPTICS', 
                'SpectralClustering', 'DBSCAN', 'MeanShift'],
                "optimizer": 'smac',
                "n_evaluations": n_loops,
                "run_obj": 'quality',
                "seed": s,
                "cutoff_time": 180,
                "preprocess_dict": {
                    "numeric_cols": list(dataset.columns),
                    "categorical_cols": [],
                    "ordinal_cols": [],
                    "y_col": [],
                },
                "evaluator": get_evaluator(evaluator_ls=['silhouetteScore'], weights=[], clustering_num=None, min_proportion=.01),
                "verbose_level": 2,
                "n_folds": 3,
                "warmstart": True,
                "warmstart_datasets_dir": 'experiments/metaknowledge/benchmark_silhouette/',
                "warmstart_metafeatures_table_path": './experiments/metaknowledge/benchmark_silhouette_metafeatures_table.csv',
                "warmstart_n_neighbors": 3,
                "warmstart_top_n": 10,
                "general_metafeatures": general_metafeatures,
                "numeric_metafeatures": numeric_metafeatures,
                "categorical_metafeatures": [],
                }
                
                cluster = AutoCluster()

                try:
                    res, predictions, running_time = fit_and_predict(cluster, dataset, dataset_name, fit_params)
                except TimeoutError:
                    print(f"Timeout: The fitting and predicting process for {dataset_name} exceeded 5 hours.")
                    # Handle the timeout as needed (e.g., log, continue with the next dataset, etc.)
                    pass


                record = cluster.get_trajectory()
                run['optimal_cfg'] = dict(res['optimal_cfg'])
                try:
                    sil = metrics.silhouette_score(dataset, predictions, metric='euclidean')
                    ari = metrics.adjusted_rand_score(true_labels, predictions)
                    dbs = metrics.davies_bouldin_score(dataset, predictions)
                    run['sil'] = sil
                    run['dbs'] = dbs
                    run['ari'] = ari
                except:
                    pass
                
                run['running_time_min'] = (round((running_time / 60), 3)) 
                result = result.append(run, ignore_index=True)
                result.to_csv('autocluster_experiments.csv', mode='a', index=False, header=False)

if __name__ == '__main__':
    run_autocluster()