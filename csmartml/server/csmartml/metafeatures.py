import numpy as np
import pandas as pd
import glob
import warnings
import scipy
import sklearn
# import patsy
from scipy.sparse import csr_matrix
from scipy.stats import kurtosis, skew, zscore
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mutual_info_score, accuracy_score, pairwise_distances
import os
# from pymfe.mfe import MFE

class Meta:

    def __init__(self, file):
        self.file = file


    def numeric_impute(self, data, num_cols, method):
        
        num_data = data[num_cols]
        if method == 'mode':
            output = num_data.fillna(getattr(num_data, method)().iloc[0])
        else:
            output = num_data.fillna(getattr(num_data, method)())
        return output

    def dict_merge(self, *args):
        imp = {}
        for dictt in args:
            imp.update(dictt)
        return imp

    def summary_stats(self, data, include_quantiles=False):
        quantiles = np.quantile(data, [0, 0.25, 0.75, 1])
        minn = quantiles[0]
        maxx = quantiles[-1]
        q1 = quantiles[1]
        q3 = quantiles[2]
        mean = np.mean(data)
        std = np.std(data)
        
        if include_quantiles:
            return minn, q1, mean, std, q3, maxx
        else:
            return minn, mean, std, maxx

    def pair_corr(self, data):    
        cors = abs(data.corr().values)
        cors = np.triu(cors, 1).flatten()
        cors = cors[cors != 0]
        return cors

    def preprocessing(self, data):
        
        x = data.iloc[:, :-1]
        # selecting the response variable
        y = data.iloc[:, -1]
        # one-hot encoding
        x = pd.get_dummies(x)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        return x, y

    def meta_features(self, data, num_cols, categorical_cols):
        
        metafeatures = dict()
        target_variable = data.iloc[:, -1]
        
        #nr_classes = target_variable.nunique()
        #metafeatures['nr_classes'] = nr_classes

        nr_instances = data.shape[0]
        #metafeatures['nr_instances'] = nr_instances
        
        # Log2 of Data Size
        log_nr_instances = np.log2(nr_instances)
        metafeatures['log_nr_instances'] = log_nr_instances
        
        nr_features = data.shape[1]
        #metafeatures['nr_features'] = nr_features
        
        # Log2 of Number of Features/Attributes
        log_nr_features = np.log(nr_features)
        metafeatures['log_nr_features'] = log_nr_features
        
        missing_val = data.isnull().sum().sum() + data.isna().sum().sum()
        metafeatures['missing_val'] = missing_val
        
        # Ratio of Missing Values 
        ratio_missing_val = missing_val / data.size
        #metafeatures['ratio_missing_val'] = ratio_missing_val
        
        # Number of Numerical Features 
        nr_numerical_features = len(num_cols)
        #metafeatures['nr_numerical_features'] = nr_numerical_features
        
        # Number of Categorical Features 
        nr_categorical_features = len(categorical_cols)
        #metafeatures['nr_categorical_features'] = nr_categorical_features

        # Dataset Ratio
        dataset_ratio = nr_features / nr_instances
        #metafeatures['dataset_ratio'] = dataset_ratio
            
        # Categorical Features Statistics
        if nr_categorical_features != 0:

            labels = data[categorical_cols].nunique()

            # Labels Sum 
            labels_sum = np.sum(labels)
            
            # Labels Mean 
            labels_mean = np.mean(labels)

            # Labels Std 
            labels_std = np.std(labels)
            
        else:
            labels_sum = 0
            labels_mean = 0
            labels_std = 0
            
        #metafeatures['labels_sum'] = labels_sum
        #metafeatures['labels_mean'] = labels_mean
        #metafeatures['labels_std'] = labels_std

        return metafeatures

    def meta_features_model_based(self, X, y):
        metafeatures = dict()

        knn4 = KNeighborsClassifier(n_neighbors=4)
        knn4.fit(X,y)
        distances, indices = knn4.kneighbors(X)

        # Average distance to 4th nearest neighbor
        metafeatures["knn4"] = np.mean(distances)

        A = knn4.kneighbors_graph(X)
        nodes = A.shape[0]
        edges = csr_matrix.getnnz(A)

        #Node to edge ratio in kNN graph Bisection
        #metafeatures["n2er"] = nodes / edges

        return metafeatures

    def meta_features_statistical(self, data, num_cols):
        
        metafeatures = dict()
        nr_numerical_features = len(num_cols)
        
        if nr_numerical_features != 0:
            
            skewness_values = abs(data[num_cols].skew())
            kurtosis_values = data[num_cols].kurtosis()        
                    
            skew_min, skew_q1, \
            skew_mean, skew_std, \
            skew_q3, skew_max = self.summary_stats(skewness_values, 
                                            include_quantiles=True)
            
            kurtosis_min, kurtosis_q1, \
            kurtosis_mean, kurtosis_std, \
            kurtosis_q3, kurtosis_max = self.summary_stats(kurtosis_values,
                                                    include_quantiles=True)
                
            pairwise_correlations = self.pair_corr(data[num_cols])
            try:
                rho_min, rho_mean, \
                rho_std, rho_max = self.summary_stats(pairwise_correlations)
            except IndexError:
                pass
                        
        var_names = ['skew_min', 'skew_std', 'skew_mean',
                    'skew_q1', 'skew_q3', 'skew_max',
                    'kurtosis_min', 'kurtosis_std', 'kurtosis_mean',
                    'kurtosis_q1', 'kurtosis_q3', 'kurtosis_max',
                    'rho_min', 'rho_max', 'rho_mean', 'rho_std']

        for var in var_names:
            try:
                metafeatures[var] = eval(var)            
            except NameError:           
                metafeatures[var] = 0
        

        return metafeatures

    def new_metafeatures(self, data):

        metafeatures = dict()
        attr_variance, attr_cv, attr_cvq1 = [], [], []

        for attr in list(data):
            temp_X = np.asarray(list(data[attr]))
            #print(np.is_na())
            attr_variance.append(np.var(temp_X))
            attr_cv.append(np.std(temp_X)/np.mean(temp_X))
            attr_cvq1.append(np.quantile(temp_X, 0.25, axis=0)/np.mean(temp_X))

        # Average attribute variance (sigma).
        metafeatures['attr_var'] = np.mean(attr_variance)

        # Coefficient of variation (CV) defined as the ratio of the standard deviation  to the attribute mean
        metafeatures['attr_cv'] = np.mean(attr_cv)
        
        # Standard deviation of all attributes first quartiles divided by their means.
        metafeatures['attr_cvq1'] = np.std(attr_cvq1)

        pca =  PCA(n_components = data.shape[1]).fit(data)

        # Variance of first principal component
        metafeatures['pca_variance'] = pca.explained_variance_[0]

        # Skewness of first principal component
        #metafeatures['pca_skewness'] = pca.components_[0].skew()

        # Dimensionality of PCA
        #metafeatures['pca_dimensionality'] = pca.transform(data).shape

        #print(pca.components_[0])

        return metafeatures

    # Compute Pairwise distances and normalize
    def distance_transform(self, data):
        pw_distance = pairwise_distances(data, metric='euclidean')
        pw_distance = normalize(pw_distance)
        return np.concatenate(pw_distance, axis=0)

    # Get percentages of normalized values within specified ranges
    def interval_percentages(self, data):

        metafeatures = dict()
        n_rows = len (data)
        md6, md7, md8, md9, md10, md11, md12, md13, md14, md15 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

        for value in data:
            if (value >= 0.0 and value <= 0.1):
                md6 += 1
            elif (value>0.1 and value<= 0.2):
                md7 += 1
            elif (value>0.2 and value<= 0.3):
                md8 += 1
            elif (value>0.3 and value<= 0.4):
                md9 += 1
            elif (value>0.4 and value<= 0.5):
                md10 += 1
            elif (value>0.5 and value<= 0.5):
                md11 += 1
            elif (value>0.6 and value<= 0.7):
                md12 += 1
            elif (value>0.7 and value<= 0.8):
                md13 += 1
            elif (value>0.8 and value<= 0.9):
                md14 += 1
            elif (value>0.9 and value<= 1.0):
                md15 += 1
            
        metafeatures["MD6"] = md6/n_rows * 100
        metafeatures["MD7"] = md7/n_rows * 100
        metafeatures["MD8"] = md8/n_rows * 100
        metafeatures["MD9"] = md9/n_rows * 100
        metafeatures["MD10"] = md10/n_rows * 100
        metafeatures["MD11"] = md11/n_rows * 100
        metafeatures["MD12"] = md12/n_rows * 100
        metafeatures["MD13"] = md13/n_rows * 100
        metafeatures["MD14"] = md14/n_rows * 100
        metafeatures["MD15"] = md15/n_rows * 100

        return metafeatures

    def zscore_interval(self, data):

        metafeatures = dict()

        md16, md17, md18, md19 = 0, 0, 0, 0
        n_rows = len(data)

        for value in zscore(data):
            if (value >= 0.0 and value < 1):
                md16 += 1
            elif (value>=1 and value<2):
                md17 += 1
            elif (value>=2 and value<3):
                md18 += 1
            elif (value>=3):
                md19 += 1
        
        metafeatures["MD16"] = md16/n_rows * 100 
        metafeatures["MD17"] = md17/n_rows * 100
        metafeatures["MD18"] = md18/n_rows * 100
        metafeatures["MD19"] = md19/n_rows * 100

        return metafeatures

    def basic_statistics(self, data):
        metafeatures = dict()
        metafeatures["mean"] = np.mean(data)
        metafeatures["sd"] = np.std(data)
        metafeatures["var"] = np.var(data)
        metafeatures["kurtosis"] = kurtosis(data)
        metafeatures["skewness"] = skew(data)

        return metafeatures


    def all_metafeatures(self, data, num_cols, metafeatures1):
        metafeatures2 = self.meta_features_statistical(data, num_cols)
        X, y = self.preprocessing(data)                             
        metafeatures3 = self.new_metafeatures(data)
        metafeatures4 = self.meta_features_model_based(X, y)
        metafeatures = self.dict_merge(metafeatures1, metafeatures2, 
                                    metafeatures3, metafeatures4)
        return metafeatures

    def all_metafeatures_db(self, data):
        metafeatures2 = self.basic_statistics(data)
        metafeatures3 = self.interval_percentages(data)
        metafeatures4 = self.zscore_interval(data)
        metafeatures = self.dict_merge(metafeatures2, metafeatures3, metafeatures4)
        return metafeatures
        

    def extract_metafeatures(self, file, meta_type, ds=None):
    
        warnings.filterwarnings("ignore")
        if ds is None:
            data = pd.read_csv("./datasets/{}.csv".format(file),
                            index_col=None,
                            header=0,
                            na_values='?')
        else:
            data = ds

        data.columns = map(str.lower, data.columns)

        # removing an id column if exists
        if 'id' in data.columns:
            data = data.drop('id', 1)

        # remove constant columns
        data = data.loc[:, (data != data.iloc[0]).any()]
        const_col = data.std().index[data.std() == 0]
        data = data.drop(const_col, axis=1)
        
        # remove columns with only NaN values
        empty_cols = ~data.isna().all()
        data = data.loc[:, empty_cols]
        
        cols = set(data.columns)
        num_cols = set(data._get_numeric_data().columns)
        categorical_cols = list(cols.difference(num_cols))

        # data imputation for categorical features
        categ_data = data[categorical_cols]
        data[categorical_cols] = categ_data.fillna(categ_data.mode().iloc[0])
        metafeatures1 = self.meta_features(data, num_cols, categorical_cols)
        
        # Numerical Features Statistics
        missing_val = metafeatures1['missing_val'] 

        if missing_val != 0:
        
            imputation_types = ['mean', 'median', 'mode']
            imputed_data = data.copy()
            results = pd.DataFrame()

            for index, num_imput_type in enumerate(imputation_types):          
                num_cols = list(num_cols) 
                
                if meta_type == "attribute":
                    imputed_data[num_cols] = self.numeric_impute(data, num_cols, num_imput_type)
                    #metafeatures1['num_imput_type'] = num_imput_type
                    metafeatures = self.all_metafeatures(imputed_data, num_cols, metafeatures1)
                elif meta_type == "distance":
                    imputed_data[num_cols] = self.numeric_impute(data, num_cols, num_imput_type)
                    metafeatures = self.all_metafeatures_db(imputed_data)
                    
                df = pd.DataFrame([metafeatures])
                results = pd.concat([results, df], axis=0)
        else:
            if meta_type == "attribute":
                #metafeatures1['num_imput_type'] = None
                metafeatures = self.all_metafeatures(data, num_cols, metafeatures1)
                results = pd.DataFrame([metafeatures])
            elif meta_type == "distance":
                metafeatures = self.all_metafeatures_db(self.distance_transform(data))
                results = pd.DataFrame([metafeatures])

        # if ds is None:
        #     dataset_name = file.split('\\')[-1]
        #     results['dataset'] = dataset_name.split('/')[3].split('.')[0]
        # else:
        results["dataset"] = file

        return results

    def extract_for_all(self, path, meta_type):
    
        allFiles = glob.glob(path + "*.csv")

        results = pd.DataFrame()
        for idx, file in enumerate(allFiles):
            d_name = file.split('//')[-1]
            print('Dataset {}({})'.format(idx + 1, d_name))
            results = pd.concat([results, self.extract_metafeatures(file, meta_type)], axis=0)

        results.to_csv('metafeatures-3.csv',
                    header=True,
                    index=False)


#print(extract_for_all("./Datasets/processed/", "distance"))

# Example usage of the Meta class
def main():
    # Path to the dataset folder
    dataset_path = "./datasets/"  # Update this path to the directory containing your datasets
    # Check if the path exists
    if not os.path.exists(dataset_path):
        print(f"The specified path '{dataset_path}' does not exist.")
        return
    meta_type = "attribute" 
    # List all files in the directory
    print("Files in the specified directory:")
    res = pd.DataFrame()
    meta = Meta(None)
    for filename in os.listdir(dataset_path):
        tmp = meta.extract_metafeatures(file=filename.split('.csv')[0], meta_type='distance')
        res = pd.concat([res, tmp], ignore_index=True)


    res.to_csv('prova.csv')
if __name__ == "__main__":
    main()
