import sys
import os
from .metafeatures import Meta

import pandas as pd 
import numpy as np
import csv
from ast import literal_eval
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from pathlib import Path

script_directory = Path(__file__).resolve().parent


class Meta_CVI:

    def __init__(self, file, meta_type, data=None):
        self.file = file
        self.meta_type = meta_type
        self.data = data

    def extract_metafeature(self):
        mf = Meta(self.file)
        if self.data is None:
            return mf.extract_metafeatures(self.file, self.meta_type)
        else:
            return mf.extract_metafeatures(self.file, self.meta_type, self.data)

    # Static method, format
    def stock(self, index):
        if index == 'iindex':
            return 'i_index'
        elif index == 'sdbw':
            return 's_dbw'
        else:
            return index

    # Static method, format
    def format_cvi(self, index):
        # Evaluation Labels
        eval_labels = {"baker_hubert_gamma": -1, "banfeld_raferty": -1, "davies_bouldin": -1, "dunns_index": 1,
                       "mcclain_rao": -1, "pbm_index": 1, "ratkowsky_lance": 1, "ray_turi": -1, "scott_symons": -1,
                       "wemmert_gancarski": 1, "xie_beni": -1, "c_index": -1, "g_plus_index": -1, "i_index": 1,
                       "modified_hubert_t": 1, "point_biserial": 1, "s_dbw": -1, "silhouette": 1, "tau_index": 1,
                       "IIndex": 1, "SDBW": -1, "calinski_harabasz_score": 1}

        # Format CVI recommendation result
        index = index.strip('[')
        index = index.strip(']')
        index = index.replace("'", "")
        index = index.split(',')

        # For 3 CVIs get measure, return required format

        cvi1 = index[0].lower().strip()
        cvi2 = index[1].lower().strip()
        cvi3 = index[2].lower().strip()

        # Final check for SDBW and IIndex

        cvi1 = self.stock(cvi1)
        cvi2 = self.stock(cvi2)
        cvi3 = self.stock(cvi3)

        return [[cvi1, eval_labels[cvi1]], [cvi2, eval_labels[cvi2]], [cvi3, eval_labels[cvi3]]]

    def search(self, algorithm ="kmeans"):

        # 1 - Get other metafeatures from knowledge-base & their CVI combinations

        df_meta_db = pd.read_csv(script_directory / "metafeatures.csv")

        df_meta_db = df_meta_db[df_meta_db.dataset != self.file]

        df_meta_instance = self.extract_metafeature()
        df_meta_db = df_meta_db.append(df_meta_instance)

        # 2 - Get known CVI combinations for datasets
        # Select pareto CVI rankings by algorithm

        filename = "multi_" + algorithm
        combinations = []
        with open(script_directory / "new-pareto" / f'{filename}.tsv') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames=['dataset', 'cvi', 'nmi'])
            for row in reader:
                combinations.append(row)

        df_combinations = pd.DataFrame(combinations)
        df_combinations['nmi'] = df_combinations['nmi'].apply(pd.np.float64)
        df_combinations['rank'] = df_combinations.groupby('dataset')['nmi'].rank(ascending=False, axis=0)
        df_combinations = df_combinations.groupby('dataset').apply(lambda x: x.sort_values(['cvi']))

        # 3 - Compute Euclidean distance between instance and other metafeatures
        df_meta_db_val = df_meta_db.loc[:, df_meta_db.columns != 'dataset']
        distance_matrix = cdist(df_meta_db_val, df_meta_db_val, metric = 'euclidean')
        
        instance_index = len(df_meta_db) - 1
        distances = np.trim_zeros(distance_matrix[instance_index])
        distances_sm = np.sort(distances)[0:5]

        # 4 - Get closest meta-features by 5-NN & merge rankings
        all_rank = []
        all_cvis = []
        for dist in distances_sm:
            index = np.where(distances == dist)
            ds = str(df_meta_db.iloc[index].dataset.values[0])
            all_rank.append(df_combinations.loc[df_combinations['dataset'] == (ds)]['rank'].values)

            if len(all_cvis) == 0:
                all_cvis = df_combinations.loc[df_combinations['dataset'] == (ds)]['cvi'].values
        
        # 5 - Select and return best CVI combination by NMI Scores
        values = []
        for rn in all_rank:
            if rn != []:
                values.append(rn)

        if len(values) > 0:
            fn_rank = np.mean(values, axis=0)       # Merged final rankings
            top_rank_index = np.min(fn_rank)        # Get highest ranked CVI combo
            fn_cvi = all_cvis[np.where(fn_rank == top_rank_index)][0]
            return self.format_cvi(fn_cvi)
        else:
            print("Can\'t recommend CVI. No correlation data for neighbors")
            # Default CVI combination
            return self.format_cvi("['SDBW', 'IIndex', 'Banfeld_Raferty']")
