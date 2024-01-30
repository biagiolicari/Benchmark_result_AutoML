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
mkr_path = Path("../src/MetaKnowledgeRepository/")

# Specify meta-feature set to use. This is the set General+Stats+Info 
mf_set = MetaFeatureExtractor.meta_feature_sets[4]