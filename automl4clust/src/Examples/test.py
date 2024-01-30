from sklearn.datasets import make_blobs
import ConfigSpace as CS

from ..Algorithm import ClusteringAlgorithms
from Metrics.MetricHandler import MetricCollection
from Optimizer.Optimizer import SMACOptimizer, RandomOptimizer, BOHBOptimizer, HyperBandOptimizer
import ConfigSpace.hyperparameters as CSH

# Create a testing data set for all examples
X, y = make_blobs(n_samples=10, n_features=2)

# optimizers that can be used in our implementation
optimizers = [RandomOptimizer, SMACOptimizer, HyperBandOptimizer, BOHBOptimizer]

##########################################################
### Example 1: Running optimizer with default settings ###
# We use Hyperband in our examples
optimizer = SMACOptimizer

# Simple example running optimizer with default settings
automl_four_clust_instance = optimizer(dataset=X)
result = automl_four_clust_instance.optimize()
best_configuration = automl_four_clust_instance.get_best_configuration()

# It is also possible to get the history of configurations
history = automl_four_clust_instance.get_config_history()

print(history)
