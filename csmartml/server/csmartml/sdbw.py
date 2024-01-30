import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans



class sdbw:
    def __init__(self, X, Y, centroids):
        self.X = X
        self.Y = Y
        self.len = len(np.bincount(Y)) #Number of clusters
        self.centroids = centroids
   
    def scatter(self):
        return self.cluster_variance() / (self.len * self.dataset_variance())
    
    def density(self):

        cluster = [self.X[self.Y == k] for k in range(self.len)]
        sigma = np.sqrt(self.cluster_variance())/self.len
        density = 0

        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for a,b in enumerate(cluster):
                    for c in [b]:
                        if i != a:
                            df_2 = pd.DataFrame(c)
                            density += self.rkk(sigma, df, df_2, self.centroids[i], self.centroids[a])

        density *= 1.0 * (self.len * (self.len - 1))            
        return density
    

    def cluster_variance(self):

        cluster = [self.X[self.Y == k] for k in range(self.len)]
        sigma, total_variance = 0, 0

        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                cluster_len = len(df)
                variance = 0
                for index, row in df.iterrows():
                    distance = euclidean(list(row), self.centroids[i])
                    variance += (distance * distance)
                sigma = variance/cluster_len
                total_variance += (sigma * sigma)

        return np.sqrt(total_variance)

   
    def dataset_variance(self):
        dataset_mean = np.mean(self.X)
        cluster = [self.X[self.Y == k] for k in range(self.len)]
        variance = 0

        for i,k in enumerate(cluster):
            for r in [k]:
                df = pd.DataFrame(r)
                for index, row in df.iterrows():
                    distance = euclidean(list(row), dataset_mean)
                    variance += (distance * distance)

        return variance/len(self.X)
    
    def gamma(self, sigma, a, b, centroid):

        distance, density = 0, 0

        for index, row in a.iterrows():
            distance = euclidean(list(row), centroid)
            if distance < sigma:
                density += 1
        
        for index, row in b.iterrows():
            distance = euclidean(list(row), centroid)
            if distance < sigma:
                density += 1
        
        return density

    def rkk(self, sigma, a, b, c_1, c_2):

        pair_centroid = (c_1 + c_2)/2
        denom = np.max([self.gamma(sigma, a, b, c_1), self.gamma(sigma, a, b, c_2)])
        res = self.gamma(sigma, a, b, pair_centroid)/denom

        return res


    def sdbw_score(self):
        return self.scatter() + self.density()