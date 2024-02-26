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

N_EXECUTIONS = 5
N_WARMSTART = 10 # Number of warmstart configurations (has to be smaller than n_loops)
N_LOOPS = 100 # Number of optimizer loops. This is n_loops = n_warmstarts + x
LIMIT_CS = True # Reduces the search space to suitable algorithms, dependening on warmstart configurations
TIME_LIMIT = 18000 # Time limit (s) of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
CVI = "predict" #predict a cvi based on our meta-knowledge

MF_INDEX_TO_USE = 5 # stats + general + theory

# Define the relative path to MetaKnowledgeRepository from the script directory
RELATIVE_PATH_MKR = "src/MetaKnowledgeRepository/"
# Define the relative path to related work from the script directory
RELATIVE_PATH_RW = "src/Experiments/RelatedWork/related_work"

# Get the directory of the current script
script_directory = Path(__file__).resolve().parent

# Set the mkr_path using the relative path
mkr_path = script_directory / RELATIVE_PATH_MKR
mf_set = MetaFeatureExtractor.meta_feature_sets[MF_INDEX_TO_USE]

related_work_path = script_directory / RELATIVE_PATH_RW

#Datasets Path
RELATIVE_PATH_TO_DATASETS = "../datasets"

#Column of interest in DF
COL_NAMES_DATAFRAME = ['dataset','framework','dbs','sil','ari','running_time_min', 'optimal_cfg']


def prepare_dataset(relative_path: Path):
    """
    Read CSV files from a given directory and prepare datasets.

    Args:
    - relative_path (Path): The relative path to the directory containing CSV files.

    Returns:
    - datasets_to_use: List of datasets.
    - true_labels_to_use: List of true labels for each dataset.
    - dataset_names_to_use: List of dataset names.
    """

    if not relative_path.is_dir():
        raise ValueError("Provided path is not a directory.")

    dataset_files = list(relative_path.glob("*.csv"))
    if not dataset_files:
        raise ValueError("No CSV files found in the directory.")
       
    datasets_to_use = []
    true_labels_to_use = []
    dataset_names_to_use = []

    for file in dataset_files:
        filename = file.stem
        data = pd.read_csv(file)
        features = data.iloc[:, :-1].values.tolist()
        labels = data.iloc[:, -1].values.tolist()

        datasets_to_use.append(features)
        true_labels_to_use.append(labels)
        dataset_names_to_use.append(filename)

    return datasets_to_use, true_labels_to_use, dataset_names_to_use

def create_dataframe_if_not_exists(file_path: str, columns: list) -> pd.DataFrame:
    """
    Create a DataFrame with specified columns if the file does not exist.

    Args:
    - file_path (str): The path to the CSV file.
    - columns (list): List of column names for the DataFrame.

    Returns:
    - DataFrame: The initial DataFrame.
    """
    if not os.path.exists(file_path):
        initial_df = pd.DataFrame(columns=columns)
        initial_df.to_csv(file_path, index=False)
        print(f"CSV file '{file_path}' created with columns: {columns}")
        return initial_df
    else:
        print(f"CSV file '{file_path}' already exists.")
        return pd.read_csv(file_path)
    
def run_ml2dac_experiment(dataset: pd.DataFrame, dataset_name: str, dataset_labels: list, mkr_path: str,
                           mf_set: str, n_loops: int, cvi: str, time_limit: int, limit_cs: bool, n_warmstarts: int):
    """
    Run ML2DAC experiment for a dataset.

    Args:
    - dataset: The dataset.
    - dataset_name (str): Name of the dataset.
    - dataset_labels (list): True labels of the dataset.
    - mkr_path (str): Path to MKR files.
    - mf_set (str): Metafeature set.
    - n_loops (int): Number of optimizer loops.
    - cvi (str): Cross-validation index.
    - time_limit (int): Time limit for optimization.
    - limit_cs (bool): Limit of candidate solutions.
    - n_warmstarts (int): Number of warm starts.

    Returns:
    - pd.Dataframe: Results of the experiment.
    """
    result = pd.DataFrame(columns=COL_NAMES_DATAFRAME)
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

    running_time = round(time.time() - t0, 2)

    online_opt_result_df = optimizer_result.get_runhistory_df()
    incumbent_stats = optimizer_result.get_incumbent_stats()
    incumbent = optimizer_result.get_incumbent()

    run['optimal_cfg'] = dict(incumbent)
    run['sil'] = metrics.silhouette_score(dataset, incumbent_stats["labels"])
    run['dbs'] = metrics.davies_bouldin_score(dataset, incumbent_stats["labels"])
    run['ari'] = metrics.adjusted_rand_score(dataset_labels, incumbent_stats["labels"])
    run['running_time_min'] = round((running_time / 60), 3)

    result = result.append(run, ignore_index=True)

    return result

def save_run_to_csv(result: pd.DataFrame, file_path: str):
    """
    Save experiment results to a CSV file.

    Args:
    - results (pd.Dataframe): DF of experiment.
    - file_path (str): Path to the CSV file.
    """
    result.to_csv(file_path, mode='a', index=False, header=False)

def run_ml2dac_experiments():
    """

    Run ML2DAC experiments for multiple datasets.

    """
    data_path = script_directory / RELATIVE_PATH_TO_DATASETS
    
    datasets_to_use, true_labels_to_use, dataset_names_to_use = prepare_dataset(data_path)
    file_path = 'ml2dac_experiments.csv'

    initial_df = create_dataframe_if_not_exists(file_path, COL_NAMES_DATAFRAME)

    for i in range(N_EXECUTIONS):
        for dataset, dataset_name, dataset_labels in zip(datasets_to_use, dataset_names_to_use, true_labels_to_use):
            result = run_ml2dac_experiment(dataset, dataset_name, dataset_labels, mkr_path, mf_set,
                                            N_LOOPS, CVI, TIME_LIMIT, LIMIT_CS, N_WARMSTART)
            save_run_to_csv(result=result, file_path=file_path)
            subprocess.run(['rm', '-rf', './smac/BO/'], check=True)


if __name__ == '__main__':
    run_ml2dac_experiments()
