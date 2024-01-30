from sklearn import cluster, mixture, manifold, decomposition
from numpy import prod
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
UniformFloatHyperparameter, UniformIntegerHyperparameter, OrdinalHyperparameter
from ConfigSpace.conditions import InCondition
from ConfigSpace import ForbiddenAndConjunction, ForbiddenEqualsClause, ForbiddenInClause

class algorithms(object):
    # this class is just to create an extra layer of namespace
    
    class Metaclass(type):
        # metaclass to ensure that static variables in the classes below are read-only
        @property
        def name(cls):
            return cls._name

        @property
        def model(cls):
            return cls._model

        @property
        def params(cls):
            return cls._params

        @property
        def params_names(cls):
            return cls._params_names
        
        @property
        def conditions(cls):
            return cls._conditions
        
        @property
        def forbidden_clauses(cls):
            return cls._forbidden_clauses
        
        @property
        def has_discrete_cfg_space(cls):
            is_discrete = lambda param: isinstance(param, UniformIntegerHyperparameter) or \
                                        isinstance(param, OrdinalHyperparameter) or \
                                        isinstance(param, CategoricalHyperparameter)
            return all([is_discrete(param) for param in cls._params])
        
        @property
        def n_possible_cfgs(cls):
            if not cls.has_discrete_cfg_space:
                return float('inf')
            else:
                def n_possible_values(param):
                    if isinstance(param, CategoricalHyperparameter):
                        return len(param.choices)
                    elif isinstance(param, OrdinalHyperparameter):
                        return len(param.sequence)
                    elif isinstance(param, UniformIntegerHyperparameter):
                        return param.upper - param.lower + 1

                return prod([n_possible_values(param) for param in cls._params])

    class DBSCAN(object, metaclass=Metaclass):
        # static variables
        _name = "DBSCAN"
        _model = cluster.DBSCAN
        _params = [
            UniformFloatHyperparameter("eps", 0.01, 5, default_value=0.01),
            UniformIntegerHyperparameter("min_samples", 5, 100, default_value=5)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []

    class KMeans(object, metaclass=Metaclass):
        # static variables
        _name = "KMeans"
        _model = cluster.KMeans
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 80, default_value=5)
            # UniformIntegerHyperparameter("random_state", 0, 9, default_value=0)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class MiniBatchKMeans(object, metaclass=Metaclass):
        # static variables
        _name = "MiniBatchKMeans"
        _model = cluster.MiniBatchKMeans
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 80, default_value=10),
            UniformIntegerHyperparameter("batch_size", 10, 1000, default_value=100),
            # UniformIntegerHyperparameter("random_state", 0, 9, default_value=0)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
    
    class AffinityPropagation(object, metaclass=Metaclass):
        # static variables
        _name = "AffinityPropagation"
        _model = cluster.AffinityPropagation
        _params = [
            UniformFloatHyperparameter("damping", 0.5, 1, default_value=0.5),
            
            # "affinity" was added
            CategoricalHyperparameter("affinity", ['euclidean'], default_value='euclidean')
            
            # 'precomputed' is excluded from "affinity"s possible values
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class MeanShift(object, metaclass=Metaclass):
        # static variables
        _name = "MeanShift"
        _model = cluster.MeanShift
        _params = [
            CategoricalHyperparameter("bin_seeding", [True, False], default_value=False),
            UniformFloatHyperparameter("bandwidth", 0.1, 50)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class SpectralClustering(object, metaclass=Metaclass):
        # static variables
        _name = "SpectralClustering"
        _model = cluster.SpectralClustering
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 80, default_value=10),
            
            # None and 'lobpcg' were excluded from eigne_solver's list of possible values
            CategoricalHyperparameter("eigen_solver", ['arpack'], default_value='arpack'),
            
            # Values 'poly', 'sigmoid', 'laplacian', 'chi2' were included,
            # 'precomputed' is excluded because it requires distance matrix input
            # 'chi2' is excluded due to "ValueError: X contains negative values.""
            CategoricalHyperparameter("affinity", ['nearest_neighbors', 'poly', 'sigmoid',\
                                                   'laplacian', 'rbf'], default_value='rbf'),
            
            # "assign_labels" was added
            CategoricalHyperparameter("assign_labels", ['kmeans','discretize'], default_value='kmeans')
            
            # -----------------------------------------------------------------
            # TODO:
            # -----------------------------------------------------------------
            # error was found when 'amg' was passed into 'eigen_solver'
            # ValueError: The eigen_solver was set to 'amg', but pyamg is not available.
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class AgglomerativeClustering(object, metaclass=Metaclass):
        # static variables
        _name = "AgglomerativeClustering"
        _model = cluster.AgglomerativeClustering
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 80, default_value=10),
            CategoricalHyperparameter("linkage", 
                                      ['ward', 'complete', 'average', 'single'], 
                                      default_value='complete'),
            CategoricalHyperparameter("affinity", 
                                      ['euclidean', 'cityblock', 
                                       'l2', 'l1', 'manhattan', 'cosine'],
                                      default_value='euclidean')
            #'ward' has been included now
            # 'precomputed' has been excluded from "affinity" s possible values because it requires 
            # a precomputed distance matrix as input from user
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = [
            ForbiddenAndConjunction(ForbiddenEqualsClause(_params[1], "ward"), 
                                    ForbiddenInClause(_params[2], ['cosine', 'cityblock', 
                                                                   'l2', 'l1', 'manhattan']))
        ]
        
    class OPTICS(object, metaclass=Metaclass):
        # static variables
        _name = "OPTICS"
        _model = cluster.OPTICS
        _params = [
            UniformIntegerHyperparameter("min_samples", 5, 1000, default_value=100),
            
            # "max_eps" may not be useful
            #UniformFloatHyperparameter("max_eps", 0.01, 10, default_value=2.0),
            
            CategoricalHyperparameter("metric", ['minkowski', 'euclidean', 
                                                 'manhattan', 'l1', 'l2', 'cosine'], default_value='minkowski'),
            CategoricalHyperparameter("cluster_method", ['xi', 'dbscan'], default_value='xi')
            
            # -----------------------------------------------------------------
            # TODO:
            # -----------------------------------------------------------------
            # some metrics, like the following, are only for boolean arrays, and will lead to infinite recursion when passed in with non-boolean data
            # 'russellrao', 'sokalmichener', 'dice', 'rogerstanimoto'
            # due to this reason, the 'metric' 
            # orginal entire list of metrics:
            # ['euclidean', 'l1', 'l2', 'manhattan',\
            #   'cosine', 'cityblock', 'braycurtis',\
            #   'canberra', 'chebyshev', 'correlation',\
            #   'hamming', 'jaccard', 'kulsinski',\
            #   'mahalanobis', 'minkowski',\
            #   'seuclidean', 'russellrao', 'sokalmichener', 'dice', 'rogerstanimoto', \
            #   'sokalsneath', 'sqeuclidean', 'yule'],\
            #
            # perhaps 'metric' should be an input from user, we don't need to optimize it at all
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class Birch(object, metaclass=Metaclass):
        # static variables
        _name = "Birch"
        _model = cluster.Birch
        _params = [
            UniformIntegerHyperparameter("n_clusters", 1, 80, default_value=5),
            
            # "branching_factor" was added
            UniformIntegerHyperparameter("branching_factor", 10, 1000, default_value=50)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class GaussianMixture(object, metaclass=Metaclass):
        # static variables
        _name = "GaussianMixture"
        _model = mixture.GaussianMixture
        _params = [
            UniformIntegerHyperparameter("n_components", 1, 80, default_value=5),
            CategoricalHyperparameter("covariance_type", ['full', 'tied', 'diag', 'spherical'], default_value='full'),
            CategoricalHyperparameter("init_params", ['kmeans', 'random'], default_value='kmeans'),
            CategoricalHyperparameter("warm_start", [True, False], default_value=False),
            # UniformIntegerHyperparameter("random_state", 0, 9, default_value=0)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
    
    
    # -----------------------------------------------------------------
    # Dimensionality Reduction Algorithms
    # -----------------------------------------------------------------
    
	# TSNE does not work yet, still debugging, do not use
    class TSNE(object, metaclass=Metaclass):
        # static variables
        _name = "TSNE"
        _model = manifold.TSNE
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2,3)), default_value=2),
            UniformFloatHyperparameter("perplexity", 1, 300, default_value=30),
            UniformFloatHyperparameter("early_exaggeration", 5.0, 20.0, default_value=12.0),
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=0)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class PCA(object, metaclass=Metaclass):
		# static variables
        _name = "PCA"
        _model = decomposition.PCA
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 4)), default_value=2),
            CategoricalHyperparameter("svd_solver", ['auto', 'full', 'arpack', 'randomized'], default_value='auto'),
            CategoricalHyperparameter("whiten", [True, False], default_value=False),
            
            # "random_state" was included, used only when "svd_solver" = 'arpack', or 'randomized'
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=0)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = [
            # InCondition(child=_params[3], parent=_params[1], values=['arpack', 'randomized'])
        ]
        _forbidden_clauses = []
    
    class IncrementalPCA(object, metaclass=Metaclass):
        # static variables
        _name = "IncrementalPCA"
        _model = decomposition.IncrementalPCA
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 4)), default_value=2),
            CategoricalHyperparameter("whiten", [True, False], default_value=False),
            UniformIntegerHyperparameter("batch_size", 10, 1000, default_value=100)
        ]
        _params_names = set([p.name for p in _params]) 
        _conditions = []
        _forbidden_clauses = []
        
    class FastICA(object, metaclass=Metaclass):
        # static variables
        _name = "FastICA"
        _model = decomposition.FastICA
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 4)), default_value=2),
            CategoricalHyperparameter("algorithm", ['parallel', 'deflation'], default_value='parallel'),
            CategoricalHyperparameter("fun", ['logcosh', 'exp','cube'], default_value='logcosh'),
            CategoricalHyperparameter("whiten", [True,False], default_value=True),
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=1)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class TruncatedSVD(object, metaclass=Metaclass):
        # static variables
        _name = "TruncatedSVD"
        _model = decomposition.TruncatedSVD
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 10)), default_value=2),
            CategoricalHyperparameter("algorithm", ['arpack','randomized'], default_value='randomized'),
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=1)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class KernelPCA(object, metaclass=Metaclass):
        # static variables
        _name = "KernelPCA"
        _model = decomposition.KernelPCA
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 10)), default_value=2),
            CategoricalHyperparameter("kernel", ['linear','poly','rbf','sigmoid','cosine'], default_value='linear'),
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=1)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class LatentDirichletAllocation(object, metaclass=Metaclass):
        # static variables
        _name = "LatentDirichletAllocation"
        _model = decomposition.LatentDirichletAllocation
        _params = [
            OrdinalHyperparameter("n_components", sequence=list(range(2, 10)), default_value=2),
            CategoricalHyperparameter("learning_method", ['batch','online'], default_value='batch'),
            # OrdinalHyperparameter("random_state", sequence=list(range(10)), default_value=1)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []
        
    class NullModel(object, metaclass=Metaclass):
        # fake model class
        class model(object):
            def __init__(self, random_state=1):
                pass
            
            def fit_transform(self, data):
                return data
            
            def transform(self, data):
                return data
        
        # static variables
        # this is a dummy class, if user chooses this algorithm, then no dimension reduction is done
        _name = "NullModel"
        _model = model
        _params = [
            OrdinalHyperparameter("random_state", sequence=list(range(3)), default_value=1)
        ]
        _params_names = set([p.name for p in _params])
        _conditions = []
        _forbidden_clauses = []