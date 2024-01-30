import pandas as pd
import numpy as np
import csv
import random
from .metafeatures import Meta
from scipy.spatial.distance import cdist
from pathlib import Path
script_directory = Path(__file__).resolve().parent

class Algorithm:

    def __init__(self, file, meta_type, data=None):
        self.file = file
        self.meta_type = meta_type
        self.data = data

    def extract_metafeature(self):
        mf = Meta(self.file)
        print(self.data)
        if self.data is None:
            return mf.extract_metafeatures(self.file, self.meta_type)
        else:
            return mf.extract_metafeatures(self.file, self.meta_type, self.data)

    def search(self, algorithm="kmeans"):

        # Get other meta-features from knowledge-base & their CVI combinations
        df_meta_db = pd.read_csv(script_directory / "metafeatures.csv")
        # ds_name = self.file.split(".")[1].split("/")[3] if self.data is None else self.file
        df_meta_db = df_meta_db[df_meta_db.dataset != self.file]

        df_meta_instance = self.extract_metafeature()
        df_meta_db = df_meta_db.append(df_meta_instance)

        filename = "multi_" + algorithm
        combinations = []
        with open(script_directory / f'new-pareto/{filename}.tsv') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=['dataset', 'cvi', 'nmi'])
            for row in reader:
                combinations.append(row)

        df_algorithms = pd.read_csv(script_directory / "algorithms.csv")
        df_algorithms['nmi'] = df_algorithms['nmi'].apply(pd.np.float64)
        df_algorithms['rank'] = df_algorithms.groupby('dataset')['nmi'].rank(ascending=False, axis=0, method='min')
        df_algorithms = df_algorithms.groupby('dataset').apply(lambda x: x.sort_values(['algorithm']))

        # Compute Euclidean distance between instance and other metafeatures
        df_meta_db_val = df_meta_db.loc[:, df_meta_db.columns != 'dataset']
        distance_matrix = cdist(df_meta_db_val, df_meta_db_val, metric='euclidean')
        
        instance_index = len(df_meta_db) - 1
        distances = np.trim_zeros(distance_matrix[instance_index])
        distances_sm = np.sort(distances)[0:5]

        # Get closest meta-features by 5-NN & merge rankings
        all_rank = []
        ls_algorithms = []
        for dist in distances_sm:
            index = np.where(distances == dist)
            ds = str(df_meta_db.iloc[index].dataset.values[0])
            all_rank.append(df_algorithms.loc[df_algorithms['dataset'] == (ds)]['rank'].values)

            if len(ls_algorithms) == 0:
                ls_algorithms = df_algorithms.loc[df_algorithms['dataset'] == (ds)]['algorithm'].values

        # Select and return best CVI combination by NMI Scores
        values = []
        for rn in all_rank:
            if rn != []:
                values.append(rn)

        if len(values) > 0:
            fn_rank = np.mean(values, axis=0)       # Merged final rankings
            top_rank_index = np.min(fn_rank)        # Get highest ranked CVI combo
            ls_sel_algorithms = ls_algorithms[np.where(fn_rank == top_rank_index)]
            
            if len(ls_sel_algorithms) > 1 :
                return random.choice(ls_sel_algorithms)
            else :
                return ls_sel_algorithms[0]
        else:
            ls_rand_algorithms = ['kmeans', 'ag', 'birch', 'spectral', 'meanshift']
            return random.choice(ls_rand_algorithms)
