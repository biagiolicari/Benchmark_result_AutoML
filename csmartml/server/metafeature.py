import numpy as np
import pandas as pd
import glob
import warnings
from scipy.stats import kurtosis, skew, zscore
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from metafeature import M

# Example usage of the Meta class
def main():
    # Path to the dataset folder
    dataset_path = "./datasets/"  # Update this path to the directory containing your datasets

    # Meta feature type
    meta_type = "attribute"  # Can be "attribute" or "distance"

    # Create an instance of the Meta class
    meta_extractor = Meta(None)

    # Extract meta-features for all datasets in the specified folder
    meta_extractor.extract_for_all(dataset_path, meta_type)
    print("Meta-feature extraction complete.")

if __name__ == "__main__":
    main()
