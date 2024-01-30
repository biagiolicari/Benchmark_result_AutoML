import os
import pandas as pd
from sklearn.ensemble import IsolationForest
from utils.stringutils import StringUtils
from utils.logutils import LogUtils
from utils.metafeatures import calculate_metafeatures, MetafeatureMapper

from itertools import cycle, islice
from sklearn import cluster, metrics, manifold, ensemble, model_selection, preprocessing
import pandas as pd
from sklearn import cluster, metrics, manifold, ensemble, model_selection, preprocessing
from preprocess_data import PreprocessedDataset
import numpy as np
dataset_path = './datasets'  # Directory containing datasets
isolation_forest_contamination = 'auto'
# Initialize an empty DataFrame to store all metafeatures
all_metafeatures = pd.DataFrame()

# Iterate over each file in the dataset directory
for filename in os.listdir(dataset_path):
    if filename.endswith('.csv'):  # Ensure it's a CSV file
        file_path = os.path.join(dataset_path, filename)
        print(f"Processing {filename}")

        # Read the dataset
        tmp = pd.read_csv(file_path)

        # Preprocessing
        preprocess_dict = {}
        preprocessed_data = PreprocessedDataset(df=tmp, 
                                                y_col=tmp.columns[-1], 
                                                numeric_cols=tmp.columns[:-1].tolist(),
                                                categorical_cols=[],
                                                ordinal_cols=[],
                                                ignore_cols=[])

        # Outlier detection
        raw_data_np = preprocessed_data.X
        predicted_labels = IsolationForest(n_estimators=100, 
                                           warm_start=True,
                                           contamination=isolation_forest_contamination
                                          ).fit_predict(raw_data_np)
        idx_np = np.where(predicted_labels == 1)
        
        # Remove outliers
        raw_data_cleaned = tmp.iloc[idx_np].reset_index(drop=True)

        # Setup file_dict for metafeatures calculation
        file_dict = {
            'numeric_cols': raw_data_cleaned.columns[:-1].tolist(),  # All columns except the last one
            'categorical_cols': [],  # Assuming no categorical columns
            'ordinal_cols': [],  # Assuming no ordinal columns
            'y_col': raw_data_cleaned.columns[-1]  # The last column
        }
        metafeature_list = MetafeatureMapper.getAllMetafeatures()
        # Calculate metafeatures here
        metafeatures = calculate_metafeatures(raw_data_cleaned, file_dict, metafeature_list)
       # Append the result to the general DataFrame
        metafeatures_df = pd.DataFrame(metafeatures, columns=metafeature_list)
        metafeatures_df['dataset'] = filename
        all_metafeatures = all_metafeatures.append(metafeatures_df)

        # You can then save or process these metafeatures as needed
# Save the final DataFrame to a CSV file
all_metafeatures.to_csv('prova.csv', index=False)
