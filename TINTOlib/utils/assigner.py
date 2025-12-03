import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import quantile_transform
from sklearn.cluster import BisectingKMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from asymmetric_greedy_search import AsymmetricGreedySearch
import pandas as pd


class AssignerFactory():
    @staticmethod
    def get_assigner(name,algorithm=None,random_state=23):
        match name:
            case 'bin':
                return BinAssigner(name)
            case 'binDigitize':
                return BinDigitizeAssigner(name)
            case 'quantile_transform':
                return QuantileAssigner(name)
            case 'PixelCentroidsAssigner':
                return PixelsCentroidsAssigner(name,algorithm,random_state)
            case 'RelevanceAssigner':
                return RelevanceAssigner(name,algorithm)

class Assigner(ABC):
    def __init__(self,name):
        self.__name = name

    def get_name(self):
        return self.__name

    def _scaled_features(self, features_coord,dim):
        min = features_coord.min(axis=0)
        max = features_coord.max(axis=0)
        scaled = (features_coord - min) / (max - min)
        return np.multiply(scaled, dim)

    @abstractmethod
    def assign(self,features_data,dim):
        raise NotImplementedError("Subclasses must implement _assing.")

class DirectAssigner(Assigner):
    def __init__(self,name):
        super().__init__(name)

    def assign(self,features_data,dim):
        return self._discretize(features_data,dim)

    @abstractmethod
    def _discretize(self,data):
        raise NotImplementedError("Subclasses must implement _discretize.")


class BinDigitizeAssigner(DirectAssigner):

    def __init__(self,name):
        super().__init__(name)

    def _discretize(self,features_data,dim):
        x_column = features_data[:, 0]
        y_column = features_data[:, 1]
        features_positions = np.stack(
            (
                np.digitize(x_column, np.linspace(np.min(x_column), np.max(x_column), dim)),
                np.digitize(y_column, np.linspace(np.min(y_column), np.max(y_column), dim)),
            ),
            axis=1
        ) - 1
        return features_positions

class BinAssigner(DirectAssigner):

    def __init__(self,name):
        super().__init__(name)

    def _discretize(self,features_data,dim):
        features_coord=self._scaled_features(features_data,dim)
        features_positions=np.floor(features_coord).astype(int)
        features_positions[:, 0][features_positions[:, 0] == dim] = dim - 1
        features_positions[:, 1][features_positions[:, 1] == dim] = dim - 1
        return features_positions


class QuantileAssigner(DirectAssigner):
    def __init__(self,name):
        super().__init__(name)

    def _discretize(self,features_data,dim):
        features_data[:, 0] = quantile_transform(features_data[:, 0,None],n_quantiles=dim,output_distribution='uniform').flatten()
        features_data[:, 1]= quantile_transform(features_data[:, 1,None],n_quantiles=dim,output_distribution='uniform').flatten()
        features_coord=self._scaled_features(features_data,dim)
        features_positions = np.floor(features_coord).astype(int)
        features_positions[:, 0][features_positions[:, 0] == dim] = dim - 1
        features_positions[:, 1][features_positions[:, 1] == dim] = dim - 1
        return features_positions



class OptimizeAssigner(Assigner):
    def __init__(self, name,scale=False,algorithm='lsa'):
        super().__init__(name)
        self.__name = name
        self.__algorithm = algorithm
        self.__scale=scale

    def assign(self,features_data,dim):
        if(self.__scale):
            features_data = self._scaled_features(features_data,dim)
        empty_pixels=self._get_empty_pixels(features_data,dim)
        cost_table=self._compute_cost_table(features_data,empty_pixels)
        row_idxs,pixels_idxs=self.__optimize(cost_table)
        return self._get_features_positions(features_data,empty_pixels,row_idxs,pixels_idxs)

    @abstractmethod
    def _get_empty_pixels(self,features_data,dim):
        raise NotImplementedError("Subclasses must implement _compute_cost_table.")

    @abstractmethod
    def _compute_cost_table(self, features_data,empty_pixels):
        raise NotImplementedError("Subclasses must implement get_features_positions.")

    @abstractmethod
    def _get_features_positions(self,features_data,empty_pixels,row_idxs,pixels_idxs):
        raise NotImplementedError("Subclasses must implement get_features_positions.")

    def __optimize(self, cost_table):
        match self.__algorithm:
            case 'lsa':
                return linear_sum_assignment(cost_table)
            case 'greedy':
                ags = AsymmetricGreedySearch(backend='numba')
                return ags.optimize(cost_table, minimize=True, shuffle=True)




class PixelsCentroidsAssigner(OptimizeAssigner):
    def __init__(self,name,algorithm,random_state=23):
        super().__init__(name,True,algorithm)
        self.__clusters_labels = None
        self.__random_state = random_state

    def _compute_cost_table(self,features_coord,empty_pixels):
        if(features_coord.shape[0]>empty_pixels.shape[0]):
            self.__clusters_labels,clusters_centers=self.__cluster_features(features_coord,empty_pixels.shape[0])
            cost_table=cdist(clusters_centers,empty_pixels[:,2:],metric='euclidean')
        else:
            cost_table=cdist(features_coord,empty_pixels[:,2:],metric='euclidean')
        cost_table=cost_table**2
        return cost_table

    def __cluster_features(self,features_coord,num_pixels):
        kmeans=BisectingKMeans(n_clusters=num_pixels,random_state=self.__random_state).fit(features_coord)
        return kmeans.labels_,kmeans.cluster_centers_

    def _get_empty_pixels(self, features_data,dim):
        image_matrix = np.zeros((dim, dim))
        empty_pixels_idxs = np.argwhere(image_matrix == 0)
        pixels_centroids = empty_pixels_idxs + 0.5
        empty_pixels = np.hstack((empty_pixels_idxs, pixels_centroids))
        return empty_pixels

    def _get_features_positions(self,features_data,empty_pixels,row_idxs,pixels_idxs):
        features_positions=np.zeros((features_data.shape[0],2))
        empty_pixels=empty_pixels[:,:2]
        if(self.__clusters_labels is not None):
            for i in range(features_data.shape[0]):
                features_positions[i]=empty_pixels[pixels_idxs[row_idxs[self.__clusters_labels[i]]]]
        else:
            features_positions[row_idxs]=empty_pixels[pixels_idxs]
        return features_positions

class RelevanceAssigner(OptimizeAssigner):
    def __init__(self,name,algorithm):
        super().__init__(name,False,algorithm)

    def _compute_cost_table(self,features_data,empty_pixels):
        self.__features_overlapping = self.__get_features_overlapping(features_data)
        cost_table = np.zeros((self.__features_overlapping.shape[0], empty_pixels.shape[0]))
        i = 0
        for index, row_o, col_o, rev in self.__features_overlapping:
            for row_d, col_d in empty_pixels:
                cost_table[int(i / (empty_pixels.shape[0])), i % (empty_pixels.shape[0])] = (abs(row_d - row_o) + abs(col_d - col_o)) * rev
                i = i + 1
        return cost_table

    def __get_features_overlapping(self,features_data):
        df_rev = pd.DataFrame(features_data, columns=['row', 'col', 'rev'])
        df_rev['count'] = df_rev.groupby(['row', 'col'], as_index=False).transform('count')
        df_rev = df_rev[df_rev['count'] > 1]
        maxidxs = df_rev.groupby(['row', 'col'])['rev'].idxmax()
        df_rev.drop(maxidxs, inplace=True)
        df_rev.drop(columns=['count'], inplace=True)
        df_rev['index'] = df_rev.index
        features_overlapping = df_rev.to_numpy()
        return features_overlapping

    def _get_empty_pixels(self, features_data,dim):
        image_matrix = np.zeros((dim, dim))
        for i, j in zip(features_data[:, 0], features_data[:, 1]):
            image_matrix[int(i), int(j)] += 1
        empty_pixels = np.argwhere(image_matrix == 0)
        return empty_pixels

    def _get_features_positions(self,features_data,empty_pixels,row_idxs,pixels_idxs):
        idxs=self.__features_overlapping[row_idxs,3].astype(int)
        features_data=features_data[:,:2]
        features_data[idxs] = empty_pixels[pixels_idxs]
        return features_data