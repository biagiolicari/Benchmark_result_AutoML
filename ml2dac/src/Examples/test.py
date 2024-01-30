# %% [markdown]
# # Examples for ML2DAC
# 
# In this notebook, we show examples on how to user our approach. Especially, how to set parameters and apply it on a custom dataset. Note that we use the MetaKnowledgeRepository (MKR) that we have created with the LearningPhase.py script. Hence, have a look at that script on how to built the MKR or how to extend it.

# %%
from MetaLearning.ApplicationPhase import ApplicationPhase
from MetaLearning import MetaFeatureExtractor
from pathlib import Path
from pandas.core.common import SettingWithCopyWarning
import warnings
warnings.filterwarnings(category=RuntimeWarning, action="ignore")
warnings.filterwarnings(category=SettingWithCopyWarning, action="ignore")
import numpy as np
np.random.seed(0)
# Specify where to find our MKR
# TODO: How to fix the path issue?
mkr_path = Path("/home/licari/AutoMLExperiments/ml2dac/src/MetaKnowledgeRepository/")

# Specify meta-feature set to use. This is the set General+Stats+Info 
mf_set = MetaFeatureExtractor.meta_feature_sets[4]

# %% [markdown]
# ## Example on a simple synthetic dataset

# %% [markdown]
# First create a simple synthetic dataset.

# %%
# Create simple synthetic dataset
from sklearn.datasets import make_blobs
# We expect the data as numpy arrays
X,y = make_blobs(n_samples=1000, n_features=10, random_state=0)

# We also use a name to describe/identify this dataset
dataset_name = "simple_blobs_n1000_f10"

# %% [markdown]
# Specify some parameter settings of our approach.

# %%
# Parameters of our approach. This can be customized
n_warmstarts = 5 # Number of warmstart configurations (has to be smaller than n_loops)
n_loops = 10 # Number of optimizer loops. This is n_loops = n_warmstarts + x
limit_cs = True # Reduces the search space to suitable algorithms, dependening on warmstart configurations
time_limit = 120 * 60 # Time limit of overall optimization --> Aborts earlier if n_loops not finished but time_limit reached
cvi = "predict" # We want to predict a cvi based on our meta-knowledge

# %% [markdown]
# Instantiate our ML2DAC approach.

# %%
ML2DAC = ApplicationPhase(mkr_path=mkr_path, mf_set=mf_set)

# %% [markdown]
# Run the optimization procedure.

# %%
optimizer_result, additional_info = ML2DAC.optimize_with_meta_learning(X, n_warmstarts=n_warmstarts,
                                                                       n_optimizer_loops=n_loops, 
                                                                       limit_cs=limit_cs,
                                                                       cvi=cvi, time_limit=time_limit,
                                                                       dataset_name=dataset_name)

# %% [markdown]
# The result contains two parts: (1) opimizer_result, which contains a history of the executed configurations in their executed order, with their runtime and the scores of the selected CVI, and (2) additional_info, which has some basic information of our meta-learning procedure, i.e., how long the meta-feature extraction took, the selected CVI, the algorithms that we used in the configuraiton space, and the dataset from the MKR that was most similar to the new dataset.

# %%
optimizer_result.get_runhistory_df()

# %%
additional_info

# %% [markdown]
# Now we retrieve the best configuration with its predicted clustering labels and compare it against the ground-truth clustering.

# %%
best_config_stats = optimizer_result.get_incumbent_stats()
best_config_stats

# %%
predicted_labels = best_config_stats["labels"]

# %%
from sklearn.metrics import adjusted_rand_score
adjusted_rand_score(predicted_labels, y)


