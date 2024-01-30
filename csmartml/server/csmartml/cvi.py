"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering

Copyright (C) 2017 Naegle Lab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import math
from scipy.spatial import distance
import numpy as np
import itertools
import re
import warnings
from sklearn import metrics
from s_dbw import S_Dbw
from .sdbw import sdbw


class Validation:
    """
    Validation is a class for calculating validation metrics on a data matrix (data), given the clustering labels in labels.
    Instantiation sets validation to NaN and a description to ''. Once a metric is performed, these are replaced (unless)
    validation did not yield a valid mathematical number, which can happen in certain cases, such as when a cluster
    consists of only one member. Such results will warn the user.

    Parameters
    ----------
    data: matrix of floats
        data matrix
    labels: list of ints
        The solution labels

    Attributes
    ----------
    validation: float
        Validation metric. NaN if error
    description: string
        A description of the validation metric

    """

    def __init__(self, data, data_raw, labels):
        self.data_matrix = data
        self.data_raw = data_raw
        self.class_label = labels
        # self.cluster_centers_ = centers
        self.validation = np.nan
        self.description = ''

    def validation_metrics_available(self):
        """
        self.validation_metrics_available() returns a dictionary, whose keys are the available validation metrics
        """
        methods = [method for method in dir(self) if callable(getattr(self, method))]
        methods.remove('validation_metrics_available')
        method_dict = {}
        for method in methods:
            if not re.match('__', method) and not re.match('_validation__', method):
                method_dict[method] = ''
        return method_dict

    def ball_hall(self):
        """
        Ball-Hall Index is the mean of the mean dispersion across all clusters
        """
        self.description = 'Mean of the mean dispersions across all clusters'
        sum_total = 0

        num_cluster = len(np.unique(self.class_label))

        # iterate through all the clusters
        for i in range(num_cluster):

            indices = [t for t, x in enumerate(self.class_label) if x == i]
            cluster_member = self.data_matrix[indices, :]

            # compute the center of the cluster
            cluster_center = np.mean(cluster_member, axis=0)

            sum_dis = 0
            # iterate through all the members
            for member in cluster_member:
                sum_dis += distance.euclidean(member, cluster_center) ** 2
            sum_total += sum_dis / len(indices)

        # compute the validation
        self.validation = sum_total / num_cluster
        return self.validation

    def banfeld_raferty(self):
        """ Banfeld-Raferty index is the weighted sum of the logarithms
         of the traces of the variance-covariance matrix of each cluster
        """
        self.description = 'Weighted sum of the logarithms of the traces of the variance-covariance matrix of each cluster'
        sum_total = 0
        num_cluster = max(self.class_label) + 1

        # iterate through all the clusters
        for i in range(num_cluster):
            sum_dis = 0
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            cluster_member = self.data_matrix[indices, :]

            # compute the center of the cluster
            cluster_center = np.mean(cluster_member, 0)

            # iterate through all the members
            for member in cluster_member:
                sum_dis += distance.euclidean(member, cluster_center) ** 2

            try:
                op = sum_dis / len(indices)
                if op <= 0:
                    # warnings.warn('Cannot calculate Banfeld_Raferty, due to an undefined value', UserWarning)
                    continue
                else:
                    sum_total += len(indices) * math.log(sum_dis / len(indices))

                # return the fitness
                self.validation = sum_total
            except:
                continue

        return self.validation

    def silhouette(self):
        """
        Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
        outside of cluster. This uses sklearn.metrics version of the Silhouette.
        """
        self.description = 'Silhouette: A combination of connectedness and compactness that measures within versus to the nearest neighbor outside a cluster. A smaller value, the better the solution'

        metric = metrics.silhouette_score(self.data_matrix, self.class_label, metric='euclidean')
        self.validation = metric
        return self.validation

    def calinski_harabasz(self):
        self.validation = metrics.calinski_harabasz_score(self.data_matrix, self.class_label)
        return self.validation

    def baker_hubert_gamma(self):
        """
        Baker-Hubert Gamma Index: A measure of compactness, based on similarity between points in a cluster, compared to similarity
        with points in other clusters
        """
        self.description = 'Gamma Index: a measure of compactness'
        splus = 0
        sminus = 0
        pairDis = distance.pdist(self.data_matrix)
        numPair = len(pairDis)
        temp = np.zeros((len(self.class_label), 2))
        temp[:, 0] = self.class_label
        vecB = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(numPair - 1):
            for j in range(i + 1, numPair):
                if vecB[i] > 0 and vecB[j] == 0:
                    # heter points smaller than homo points
                    if pairDis[i] < pairDis[j]:
                        splus += 1
                    # heter points larger than homo points
                    if pairDis[i] > vecB[j]:
                        sminus += 1
                if vecB[i] == 0 and vecB[j] > 0:
                    # heter points smaller than homo points
                    if pairDis[j] < pairDis[i]:
                        splus += 1
                    # heter points larger than homo points
                    if pairDis[j] > vecB[i]:
                        sminus += + 1
        # compute the fitness
        self.validation = (splus - sminus) / (splus + sminus)
        return self.validation

    def det_ratio(self):
        """
        The determinant ratio index, a measure of connectedness
        """
        # compute the attributes number and cluster number
        self.description = 'Determinant ratio, a measure of connectedness'
        attributes = len(self.data_matrix[0])
        xData = self.data_matrix
        wg = np.zeros((attributes, attributes))
        numCluster = max(self.class_label) + 1
        # compute cluster scatter matrix
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            xCluster = clusterMember
            # iterate through attributes
            for j in range(attributes):
                columnVec = clusterMember[:, j]
                columnCenter = np.mean(columnVec)
                # compute xk
                xCluster[:, j] = columnVec - columnCenter
            # add to wg
            wg += np.dot(np.transpose(xCluster), xCluster)
        # compute data scatter matrix
        for i in range(attributes):
            columnVec = self.data_matrix[:, i]
            columnCenter = np.mean(columnVec)
            # data scatter matrix
            xData[:, i] = columnVec - columnCenter

        t = np.dot(np.transpose(xData), xData)
        # compute the fitness
        self.validation = np.linalg.det(t) / np.linalg.det(wg)
        return self.validation

    def c_index(self):
        """
        The C-Index, a measure of compactness
        """
        self.description = 'The C-Index, a measure of cluster compactness'
        sw = 0
        nw = 0
        numCluster = max(self.class_label) + 1
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute distance of every pair of points
            list_clusterDis = distance.pdist(clusterMember)
            sw = sw + sum(list_clusterDis)
            nw = nw + len(list_clusterDis)
        # compute the pairwise distance of the whole dataset
        list_dataDis = distance.pdist(self.data_matrix)
        # compute smin
        sortedList = sorted(list_dataDis)
        smin = sum(sortedList[0:nw])
        # compute smax
        sortedList = sorted(list_dataDis, reverse=True)
        smax = sum(sortedList[0:nw])
        # compute the fitness
        self.validation = (sw - smin) / (smax - smin)
        return self.validation

    def g_plus_index(self):
        """
        The G_plus index, the proportion of discordant pairs among all the pairs of distinct point, a measure of connectedness
        """
        self.description = "The G_plus index, a measure of connectedness"
        sminus = 0
        pairDis = distance.pdist(self.data_matrix)
        numPair = len(pairDis)
        temp = np.zeros((len(self.class_label), 2))
        temp[:, 0] = self.class_label
        vecB = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(numPair - 1):
            for j in range(i + 1, numPair):
                if vecB[i] > 0 and vecB[j] == 0:
                    # heter points larger than homo points
                    if pairDis[i] > vecB[j]:
                        sminus = sminus + 1
                if vecB[i] == 0 and vecB[j] > 0:
                    # heter points larger than homo points
                    if pairDis[j] > vecB[i]:
                        sminus = sminus + 1
        # return fitness
        self.validation = 2 * sminus / (numPair * (numPair - 1))
        return self.validation

    def ksq_detw_index(self):
        """
        The Ksq_DetW Index, a measure of connectedness
        """
        self.description = "The Ksq_DetW index, a measure of connectedness"
        # compute the attributes number and cluster number
        attributes = len(self.data_matrix[0])
        wg = np.zeros((attributes, attributes))
        numCluster = max(self.class_label) + 1
        # compute cluster scatter matrix
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            xCluster = clusterMember
            # iterate through attributes
            for j in range(attributes):
                columnVec = clusterMember[:, j]
                columnCenter = np.mean(columnVec)
                # compute xk
                xCluster[:, j] = columnVec - columnCenter
            # add to wg
            wg += np.dot(np.transpose(xCluster), xCluster)
        # compute fitness
        self.validation = math.pow(numCluster, 2) * np.linalg.det(wg)
        return self.validation

    def log_det_ratio(self):
        """
        The log determinant ratio index, a measure of connectedness
        """
        self.description = "The log determinant ratio index, a measure of connectedness"
        numObj = len(self.class_label)
        self.validation = numObj * math.log(self.det_ratio())
        return self.validation

    def log_ss_ratio(self):
        """
        The log ss ratio, a measure of connectedness
        """
        self.description = "The log ss ratio, a measure of connectedness"
        bgss = 0
        wgss = 0
        numCluster = max(self.class_label) + 1
        # compute the dataset center
        dataCenter = np.mean(self.data_matrix, 0)
        # iterate through the cluster
        for i in range(numCluster):
            sumTemp = 0
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # add to bgss
            bgss += len(indices) * math.pow(distance.euclidean(clusterCenter, dataCenter), 2)
            # iterate through all the members of the cluster
            for member in clusterMember:
                sumTemp += math.pow(distance.euclidean(member, clusterCenter), 2)
            wgss += sumTemp
        # compute the fitness
        self.validation = math.log(bgss / wgss)
        return self.validation

    def mcclain_rao(self):
        """
        The McClain-Rao Index, a measure of compactness
        """
        self.description = "The McClain-Rao Index, a measure of compactness"
        sw = 0
        sb = 0
        nw = 0
        numObj = len(self.class_label)
        numCluster = max(self.class_label) + 1
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute pairwise distance
            pairDis = distance.pdist(clusterMember)
            # add to sw and nw
            sw += sum(pairDis)
            nw += len(pairDis)
            # iterate the clusters again for between-cluster distance
            for j in range(numCluster):
                if j > i:
                    indices2 = [t for t, x in enumerate(self.class_label) if x == j]
                    clusterMember2 = self.data_matrix[indices2, :]
                    betweenDis = distance.cdist(clusterMember, clusterMember2)
                    # add to sb
                    sb += sum(list(itertools.chain(*betweenDis)))
        # compute nb
        nb = numObj * (numObj - 1) / 2 - nw
        # compute fitness
        self.validation = nb * sw / (nw * sb)
        return self.validation

    def pbm_index(self):
        """
        The PBM index, a measure of compactness
        """
        self.description = "The PBM index, a measure of compactness"
        ew = 0
        et = 0
        list_centerDis = []
        numCluster = max(self.class_label) + 1
        # compute the center of the dataset
        dataCenter = np.mean(self.data_matrix, 0)
        # iterate through the  clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # compute the center distance
            list_centerDis.append(distance.euclidean(dataCenter, clusterCenter))
            # iterate through the member of the  cluster
            for member in clusterMember:
                ew += distance.euclidean(member, clusterCenter)
                et += distance.euclidean(member, dataCenter)
        db = max(list_centerDis)
        # compute the fitness
        self.validation = math.pow(et * db / (numCluster * ew), 2)
        return self.validation

    def point_biserial(self):
        """
        The Point-Biserial index, a measure of connectedness
        """
        self.description = "The Point-Biserial index, a measure of connectedness"
        sw = 0
        sb = 0
        nw = 0
        numObj = len(self.class_label)
        numCluster = max(self.class_label) + 1
        nt = numObj * (numObj - 1) / 2
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute pairwise distance
            pairDis = distance.pdist(clusterMember)
            # add to sw and nw
            sw += sum(pairDis)
            nw += len(pairDis)
            # iterate the clusters again for between-cluster distance
            for j in range(numCluster):
                if j > i:
                    indices2 = [t for t, x in enumerate(self.class_label) if x == j]
                    clusterMember2 = self.data_matrix[indices2, :]
                    betweenDis = distance.cdist(clusterMember, clusterMember2)
                    # add to sb
                    sb += sum(list(itertools.chain(*betweenDis)))
        # compute nb
        nb = nt - nw
        # compute fitness
        self.validation = ((sw / nw - sb / nb) * math.sqrt(nw * nb)) / nt
        return self.validation

    def ratkowsky_lance(self):
        """
        The Ratkowsky-Lance index, a measure of compactness
        """
        self.description = "The Ratkowsky-Lance index, a measure of compactness"
        list_divide = []
        attributes = len(self.data_matrix[0])
        numCluster = max(self.class_label) + 1
        # iterate through the attributes
        for i in range(attributes):
            bgssj = 0
            tssj = 0
            columnVec = self.data_matrix[:, i]
            columnCenter = np.mean(columnVec)
            # compute bgssj
            for j in range(numCluster):
                indices = [t for t, x in enumerate(self.class_label) if x == j]
                columnCluster = self.data_matrix[indices, :]
                centerCluster = np.mean(columnCluster)
                bgssj += len(indices) * math.pow(centerCluster - columnCenter, 2)
            # iterate through the  members of the column
            for member in columnVec:
                tssj += math.pow(member - columnCenter, 2)
            list_divide.append(bgssj / tssj)
        r = sum(list_divide) / attributes
        # compute the  fitness
        self.validation = math.sqrt(r / numCluster)
        return self.validation

    def ray_turi(self):
        """
        The Ray-Turi index, a measure of compactness
        """
        self.description = "The Ray-Turi index, a measure of compactness"
        wgss = 0
        list_centers = []
        numCluster = max(self.class_label) + 1
        numObj = len(self.class_label)

        for i in range(numCluster):
            wgssk = 0
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # print(np.squeeze(np.asarray(clusterCenter)))
            list_centers.append(np.asarray(clusterCenter))
            # iterate through the  cluster members
            for member in clusterMember:
                wgssk += math.pow(distance.euclidean(member, clusterCenter), 2)
            # add to wgsss
            wgss += wgssk
        # compute the min center dis
        list_centers = np.concatenate(list_centers, axis=0)
        minDis = math.pow(np.min(distance.pdist(list_centers)), 2)
        # compute the fitness
        self.validation = wgss / (numObj * minDis)
        return self.validation

    def scott_symons(self):
        """
        The Scott-Symons index, a measure of connectedness
        """
        self.description = "The Scott-Symons index, a measure of connectedness"
        fitness = 0
        # compute the attributes number and cluster number
        attributes = len(self.data_matrix[0])
        numCluster = max(self.class_label) + 1
        # compute cluster scatter matrix
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            nk = len(indices)
            clusterMember = self.data_matrix[indices, :]
            xCluster = clusterMember
            # iterate through attributes
            for j in range(attributes):
                columnVec = clusterMember[:, j]
                columnCenter = np.mean(columnVec)
                # compute xk
                xCluster[:, j] = columnVec - columnCenter
            # compute wgk
            wgk = np.dot(np.transpose(xCluster), xCluster)
            if np.linalg.det(wgk / nk) != 0:
                fitness += nk * math.log(np.linalg.det(wgk / nk))
            else:
                warnings.warn('Cannot calculate Scott_Symons, due to an undefined value', UserWarning)
        # return fitness
        self.validation = fitness
        return self.validation

    def tau_index(self):
        """
        The Tau index, a measure of compactness
        """
        self.description = "The Tau index, a measure of compactness"
        # compute nb,nw,nt
        nw = 0
        numObj = len(self.class_label)
        numCluster = max(self.class_label) + 1
        nt = numObj * (numObj - 1) / 2
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            nk = len(indices)
            nw += nk * (nk - 1) / 2
        nb = nt - nw
        # compute s+ and s-
        splus = 0
        sminus = 0
        pairDis = distance.pdist(self.data_matrix)
        numPair = nt
        temp = np.zeros((len(self.class_label), 2))
        temp[:, 0] = self.class_label
        vecB = distance.pdist(temp)
        # iterate through all the pairwise comparisons
        for i in range(int(numPair - 1)):
            for j in range(i + 1, int(numPair)):
                if vecB[i] > 0 and vecB[j] == 0:
                    # heter points smaller than homo points
                    if pairDis[i] < pairDis[j]:
                        splus += 1
                    # heter points larger than homo points
                    if pairDis[i] > vecB[j]:
                        sminus += 1
                if vecB[i] == 0 and vecB[j] > 0:
                    # heter points smaller than homo points
                    if pairDis[j] < pairDis[i]:
                        splus += 1
                    # heter points larger than homo points
                    if pairDis[j] > vecB[i]:
                        sminus += 1
        # compute the fitness
        self.validation = (splus - sminus) / math.sqrt(nb * nw * nt * (nt - 1) / 2)
        return self.validation

    def trace_w(self):
        """
        The Trace_W index, a measure of connectedness
        """
        self.description = "The Trace_W index, a measure of connectedness"
        wgss = 0
        numCluster = max(self.class_label) + 1
        for i in range(numCluster):
            wgssk = 0
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # iterate through the  cluster members
            for member in clusterMember:
                wgssk += math.pow(distance.euclidean(member, clusterCenter), 2)
            # add to wgsss
            wgss += wgssk
        # return the fitness
        self.validation = wgss
        return self.validation

    def trace_wib(self):
        """
        The Trace_WiB index, a measure of connectedness
        """
        self.description = "The Trace_WiB index, a measure of connectedness"
        numCluster = max(self.class_label) + 1
        attributes = len(self.data_matrix[0])
        b = np.zeros((numCluster, attributes))
        wg = np.zeros((attributes, attributes))
        # compute the data center
        dataCenter = np.mean(self.data_matrix, 0)
        # iterate all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            xCluster = clusterMember
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            b[i, :] = clusterCenter - dataCenter
            # compute wgk and wg
            for j in range(attributes):
                columnVec = clusterMember[:, j]
                columnCenter = np.mean(columnVec)
                # compute xk
                xCluster[:, j] = columnVec - columnCenter
            # compute wgk
            wg += np.dot(np.transpose(xCluster), xCluster)
        # compute bg
        bg = np.dot(np.transpose(b), b)
        # compute fitness
        try:
            self.validation = np.trace(np.dot(np.linalg.inv(wg), bg))
        except np.linalg.linalg.LinAlgError:
            # Numpy will thrown an exception on singular matricies
            # If this happens, warn the user and return 0
            warnings.warn('Cannot calculate trace_wib, due to an undefined value', UserWarning)
            self.validation = 0
        return self.validation

    def wemmert_gancarski(self):
        """
        The Wemmert-Gancarski index, the quotients of distances between the points and the barycenters of all clusters, a measure of compactness
        """
        self.description = "The Wemmert-Gancarski index, a measure of compactness"
        sum = 0
        list_centers = []
        attributes = len(self.data_matrix[0])
        numObj = len(self.class_label)
        numCluster = max(self.class_label) + 1
        # compute all the centers
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            list_centers.append(np.mean(clusterMember, 0))
        # iterate the clusters again for Rm
        for i in range(numCluster):
            sumRm = 0
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the currrent center
            clusterCenter = np.mean(clusterMember, 0)
            tempList = list_centers
            tempList = tempList[:i] + tempList[i + 1:]
            # iterate through the member and compute rm
            for member in clusterMember:
                # make it a 2d array
                memberArray = np.zeros((1, attributes))
                memberArray[0, :] = member
                # compute the pair wise distance
                list_dis = distance.cdist(memberArray, tempList)
                sumRm += (distance.euclidean(member, clusterCenter)) / min(min(list_dis))
            # compute the sum
            sum += max([0, len(indices) - sumRm])
        # compute the fitness
        self.validation = sum / numObj
        return self.validation

    def root_mean_square(self):
        """
        The Root-Mean-Square Standard Deviation (RMSSTD), the root mean square
        standard deviation of all variables within each cluster. A measure of
        connectedness.
        """
        self.description = "The Root-Mean-Square Standard Deviation (RMSSTD), a measure of connectedness"
        numCluster = max(self.class_label) + 1
        attributes = len(self.data_matrix[0])
        denominator = attributes * (len(self.data_matrix) - numCluster)
        normSum = 0
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # compute the square error for every member in the cluster
            for member in clusterMember:
                normSum += distance.euclidean(member, clusterCenter)
        self.validation = math.sqrt(normSum / denominator)
        return self.validation

    def r_squared(self):
        """
        R-squared, a statistical measure of how close the data is to a fitted regression line.
        A measure of compactness.
        """
        self.description = "R-squared, a measure of compactness"
        # compute the center of the dataset
        dataCenter = np.mean(self.data_matrix, 0)
        numCluster = max(self.class_label) + 1
        normClusterSum = 0
        normDatasetSum = 0
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            # compute the norm for every member in the cluster with cluster center and dataset center
            for member in clusterMember:
                normClusterSum += distance.euclidean(member, clusterCenter)
                normDatasetSum += distance.euclidean(member, dataCenter)
        # compute the fitness
        self.validation = (normDatasetSum - normClusterSum) / normDatasetSum
        return self.validation

    def modified_hubert_t(self):
        """
        The Modified Hubert T Statistic, a measure of compactness.
        """
        self.description = "The Modified Hubert T Statistic, a measure of compactness"
        sumDiff = 0
        # compute the centers of all the clusters
        list_center = []
        numCluster = max(self.class_label) + 1
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            list_center.append(np.mean(clusterMember, 0))
        size = len(self.class_label)
        # iterate through each of the two pairs exhaustively
        for i in range(size - 1):
            for j in range(i + 1, size):
                # get the cluster labels of the two objects
                label1 = self.class_label[i]
                label2 = self.class_label[j]
                # compute the distance of the two objects
                pairDistance = distance.euclidean(self.data_matrix[i], self.data_matrix[j])
                # compute the distance of the cluster center of the two objects
                centerDistance = distance.euclidean(list_center[label1], list_center[label2])
                # add the product to the sum
                sumDiff += pairDistance * centerDistance
        # compute the fitness
        self.validation = 2 * sumDiff / (size * (size - 1))
        return self.validation

    def i_index(self):
        """
        The I index, a measure of compactness.
        """
        self.description = "The I Index, a measure of compactness."
        normClusterSum = 0
        normDatasetSum = 0
        list_centers = []
        # compute the number of clusters and attribute
        attributes = len(self.data_matrix[0])
        numCluster = max(self.class_label) + 1
        # compute the center of the dataset
        dataCenter = np.mean(self.data_matrix, 0)
        # iterate through all the clusters
        for i in range(numCluster):
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the center of the cluster
            clusterCenter = np.mean(clusterMember, 0)
            list_centers.append(np.asarray(clusterCenter))
            # compute the norm for every member in the cluster with cluster center and dataset center
            for member in clusterMember:
                normClusterSum += distance.euclidean(member, clusterCenter)
                normDatasetSum += distance.euclidean(member, dataCenter)
        # compute the max distance between cluster centers
        list_centers = np.concatenate(list_centers, axis=0)
        maxCenterDis = max(distance.pdist(list_centers))
        # compute the fitness
        self.validation = math.pow(((normDatasetSum * maxCenterDis) / (normClusterSum * numCluster)), attributes)
        return self.validation

    def davies_bouldin(self):
        """
        The Davies-Bouldin index, the average of all cluster similarities.
        """
        self.description = "The Davies-Bouldin index, the average of all cluster similarities"
        numCluster = max(self.class_label) + 1
        list_max = []
        # iterate through the clusters
        for i in range(numCluster):
            list_tempMax = []
            # get all members from cluster i
            indices1 = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember1 = self.data_matrix[indices1, :]
            # compute the cluster center
            clusterCenter1 = np.mean(clusterMember1, 0)
            # compute the cluster norm sum
            sumNorm1 = 0
            for member in clusterMember1:
                sumNorm1 += distance.euclidean(member, clusterCenter1)
            for j in range(numCluster):
                if j != i:
                    # get all members from cluster j
                    indices2 = [t for t, x in enumerate(self.class_label) if x == j]
                    clusterMember2 = self.data_matrix[indices2, :]
                    # compute the cluster center
                    clusterCenter2 = np.mean(clusterMember2, 0)
                    # compute the cluster norm sum
                    sumNorm2 = 0
                    for member in clusterMember2:
                        sumNorm2 += distance.euclidean(member, clusterCenter2)
                    tempDis = (sumNorm1 / len(indices1) + sumNorm2 / len(indices2)) / distance.euclidean(clusterCenter1,
                                                                                                         clusterCenter2)
                    list_tempMax.append(tempDis)
            list_max.append(max(list_tempMax))
        # compute the fitness
        self.validation = sum(list_max) / numCluster
        return self.validation

    def xie_beni(self):
        """
        The Xie-Beni index, a measure of compactness.
        """
        self.description = "The Xie-Beni index, a measure of compactness"
        numCluster = max(self.class_label) + 1
        numObject = len(self.class_label)
        sumNorm = 0
        list_centers = []
        for i in range(numCluster):
            # get all members from cluster i
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the cluster center
            clusterCenter = np.mean(clusterMember, 0)
            list_centers.append(np.asarray(clusterCenter))
            # interate through each member of the cluster
            for member in clusterMember:
                sumNorm += math.pow(distance.euclidean(member, clusterCenter), 2)
        list_centers = np.concatenate(list_centers, axis=0)
        minDis = min(distance.pdist(list_centers))
        # compute the fitness
        self.validation = sumNorm / (numObject * pow(minDis, 2))
        return self.validation

    ## density function for SDBW
    @staticmethod
    def __density(a, b, stdev):
        dis = distance.euclidean(a, b)
        if dis > stdev:
            return 0
        else:
            return 1

    # S_Dbw validity index SDBW
    
    # def SDBW(self):
    #     sdbw_c = sdbw(self.data_raw, self.class_label, self.cluster_centers_)
    #     self.validation = sdbw_c.sdbw_score()
    #     return self.validation

    def s_dbw(self):
        score = S_Dbw(np.asarray(self.data_raw), self.class_label, centers_id=None, method='Halkidi', alg_noise='bind',centr='mean', nearest_centr=True, metric='euclidean')
        self.validation = score
        return self.validation
    '''
    
    def s_dbw(self):
        """
        The S_Dbw index, a measure of compactness.
        """
        self.description = "The S_Dbw index, a measure of compactness"
        sumDens = 0
        sumNormCluster = 0
        sumScat = 0
        list_centers = []
        temp_c1, temp_c2 = [], []
        numCluster = max(self.class_label) + 1
        # compute the norm of sigma(dataset)
        normSigDataset = np.linalg.norm(np.var(self.data_matrix, 0))
        # iterate through all the clusters of self.class_label
        for i in range(numCluster):
            # get all members from cluster i
            indices = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember = self.data_matrix[indices, :]
            # compute the cluster center
            clusterCenter = np.mean(clusterMember, 0)
            list_centers.append(clusterCenter)
            normSigCluster = np.linalg.norm(np.var(clusterMember, 0))
            sumScat += normSigCluster / normSigDataset
            sumNormCluster += normSigCluster
        # compute stdev
        stdev = math.sqrt(sumNormCluster) / numCluster
        # iterate again for density_bw
        for i in range(numCluster):
            sumDensity1 = 0
            sumTemp = 0
            # get all members from cluster i
            indices1 = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember1 = self.data_matrix[indices1, :]
            temp_c1 = clusterMember1.copy()
            # compute sum of f(x,ci)
            for member in clusterMember1:
                sumDensity1 += Validation.__density(member, list_centers[i], stdev)
            for j in range(numCluster):
                if j != i:
                    sumDensity2 = 0
                    sumDensityCombine = 0
                    # get all members from cluster j
                    indices2 = [t for t, x in enumerate(self.class_label) if x == j]
                    clusterMember2 = self.data_matrix[indices2, :]
                    temp_c2 = clusterMember2.copy()
                    # compute sum of f(x,cj)
                    for member in clusterMember2:
                        sumDensity2 += Validation.__density(member, list_centers[j], stdev)
                    # compute the middle point of the two cluster centers
                    midPoint = []
                    for k in range(len(list_centers[0])):
                        midPoint.append(np.asarray(list_centers[i][k] + list_centers[j][k]) / 2)
                    # compute sum of f(x,uij)
                    midPoint = np.concatenate(midPoint, axis=0)
                    combined = np.concatenate((temp_c1, temp_c2))
                    for member in combined:
                        sumDensityCombine += Validation.__density(member, midPoint, stdev)
                    sumTemp += sumDensityCombine / max([sumDensity1, sumDensity2])
            sumDens += sumTemp
        # compute scat and dens_bw
        scat = sumScat / numCluster
        dens_bw = sumDens / (numCluster * (numCluster - 1))
        # compute the fitness
        self.validation = scat + dens_bw
        return self.validation
    '''
    

    def dunns_index(self):
        """
        Dunn's index, a measure of cluster compactness
        """
        self.description = "Dunn's Index, a measure of compactness"
        list_diam = []
        list_minDis = []
        numCluster = max(self.class_label) + 1
        # iterate through the clusters
        for i in range(numCluster - 1):
            # get all members from cluster i
            indices1 = [t for t, x in enumerate(self.class_label) if x == i]
            clusterMember1 = self.data_matrix[indices1, :]
            # compute the diameter of the cluster
            list_diam.append(max(distance.pdist(clusterMember1)))
            for j in range(i + 1, numCluster):
                # get all members from cluster j
                indices2 = [t for t, x in enumerate(self.class_label) if x == j]
                clusterMember2 = self.data_matrix[indices2, :]
                # compute the diameter of the cluster
                diameter = distance.pdist(clusterMember2)
                # If it is zero, the value is undefined
                if len(diameter) == 0:
                    warnings.warn('Cannot calculate Dunns_index, due to an undefined value', UserWarning)
                    self.validation = 0
                    return self.validation
                list_diam.append(max(diameter))
                # get the pairwise distance and find the minimum
                pairDis = distance.cdist(clusterMember1, clusterMember2)
                minDis = min(list(itertools.chain(*pairDis)))
                list_minDis.append(minDis)
        # compute the fitness
        return min(list_minDis) / max(list_diam)

    def run_all(self):
        metric_scores = {}
        keys = list(self.validation_metrics_available().keys())

        for key in keys:
            metric_scores[key] = "none"

            if key in ["baker_hubert_gamma", "tau_index", "g_plus_index", "log_ss_ratio", "log_det_ratio", "det_ratio",
                       "run_all", "ksq_detw_index", "modified_hubert_t"]:
                continue

            try:
                # print(key)
                metric_scores[key] = eval("self." + key + "()")

            except:
                pass

        return metric_scores

    def run_list(self, cvi_list=["banfeld_raferty", "modified_hubert_t", "point_biserial", "davies_bouldin", ""]):
        metric_scores = {}
        for key in cvi_list:
            metric_scores[key] = eval("self." + key + "()")

        return metric_scores
