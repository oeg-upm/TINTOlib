import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import quantile_transform
from sklearn.cluster import BisectingKMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from asymmetric_greedy_search import AsymmetricGreedySearch
import pandas as pd
import TINTOlib.utils.constants as constants

class AssignerFactory():
    @staticmethod
    def get_assigner(name,algorithm=None,random_state=23):
        """

        Args:
            name: Name of assigner to retrieve
            algorithm: Using in assigners that mapping each pixel with a unique feature. Allow use optimal algorithm or greedy algorithm
            random_state: seed to replicate process

        Returns:
            Assigner to mapping pixel and features
        """
        match name:
            case constants.bin_assigner:
                return BinAssigner(name)
            case constants.bin_digitize_assigner:
                return BinDigitizeAssigner(name)
            case constants.quantile_assigner:
                return QuantileAssigner(name)
            case constants.pixel_centroids_assigner:
                return PixelsCentroidsAssigner(name,algorithm,random_state)
            case constants.relevance_assigner:
                return RelevanceAssigner(name,algorithm)

class Assigner(ABC):
    def __init__(self,name):
        self.__name = name

    def get_name(self):
        return self.__name

    def _scaled_features(self, features_coord,dim):
        """
        Scale features coordinates to range [0,dim]
        Args:
            features_coord: 2D Array features coordinates
            dim: Max value of features coordinates

        Returns:
        2D Array features coordinates scaled
        """
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
        """
            Discretize features coordinates into bins using image dimension
        Args:
            features_data:features coordinates
            dim:image dim

        Returns:
            Array with image features positions
        """
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
        """
            Scale features coordinates and discretize into bins using image dimension
        Args:
            features_data:features coordinates
            dim:image dim

        Returns:
            Array with image features positions
        """
        features_coord=self._scaled_features(features_data,dim)
        features_positions=np.floor(features_coord).astype(int)
        features_positions[:, 0][features_positions[:, 0] == dim] = dim - 1
        features_positions[:, 1][features_positions[:, 1] == dim] = dim - 1
        return features_positions


class QuantileAssigner(DirectAssigner):
    def __init__(self,name):
        super().__init__(name)

    def _discretize(self,features_data,dim):
        """
              Transform features coordinates in uniform distribution, Scale and discretize into bins using image dimension
          Args:
              features_data:features coordinates
              dim:image dim

          Returns:
              Array with image features positions
          """
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
        """
            Optional Scale features coordinates, compute cost table for each mapping combinations between pixels and features and use an algorithm (optimal/suboptimal) to
            optimize a function mapping pixel and features
        Args:
            features_data: Features coordinates/positions and/or relevance by feature
            dim:image dim

        Returns:
        Array with image features positions
        """
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
        """
        Get a cost table for optimize using lsa or greedy algorithm
        Args:
            cost_table: Table with the cost of assign each pixel to each feature

        Returns:
            Pixel-feature combinations optimized
        """
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
        """
        Create cost table for assigning each pixel to each feature. Using distance between features coord and pixel centroids. If the number of pixels available is
        lower than the number of features to mapping use a cluster algorithm to group features
        Args:
            features_coord:features coordinates
            empty_pixels:Array with empty pixels positions and centroids

        Returns:
            Cost table for assigning each pixel to each feature
        """
        if(features_coord.shape[0]>empty_pixels.shape[0]):
            self.__clusters_labels,clusters_centers=self.__cluster_features(features_coord,empty_pixels.shape[0])
            cost_table=cdist(clusters_centers,empty_pixels[:,2:],metric='euclidean')
        else:
            cost_table=cdist(features_coord,empty_pixels[:,2:],metric='euclidean')
        cost_table=cost_table**2
        return cost_table

    def __cluster_features(self,features_coord,num_pixels):
        """
        Cluster features based on number of pixels
        Args:
            features_coord:features coordinates
            num_pixels:Number of pixels to cluster coordinates

        Returns:
        Cluster by feature and clusters centers
        """
        kmeans=BisectingKMeans(n_clusters=num_pixels,random_state=self.__random_state).fit(features_coord)
        return kmeans.labels_,kmeans.cluster_centers_

    def _get_empty_pixels(self, features_data,dim):
        """
        Create structure with empty pixels according to dimension and centroids
        Args:
            features_data: features information
            dim:image dim

        Returns:
        Array with empty pixels positions and centroids
        """
        image_matrix = np.zeros((dim, dim))
        empty_pixels_idxs = np.argwhere(image_matrix == 0)
        pixels_centroids = empty_pixels_idxs + 0.5
        empty_pixels = np.hstack((empty_pixels_idxs, pixels_centroids))
        return empty_pixels

    def _get_features_positions(self,features_data,empty_pixels,row_idxs,pixels_idxs):
        """
        Get features positions array according to features coordinates, empty pixels array and mapping between pixel and features optimized
        Args:
            features_data: features coordinates
            empty_pixels: Array with empty pixels positions and centroids
            row_idxs: Index of features/clusters
            pixels_idxs: Index of pixels

        Returns:
        Array with image features positions
        """
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
        """
        Create cost table for assigning each pixel to each feature that need to be relocate. Using the features relevance and the number of pixels
         between feature origin pixel assigned and each pixel available.
        Args:
            features_data:features original positions
            empty_pixels:Array with empty pixels positions

        Returns:
            Cost table for assigning each pixel to each feature
        """
        self.__features_overlapping = self.__get_features_overlapping(features_data)
        cost_table = np.zeros((self.__features_overlapping.shape[0], empty_pixels.shape[0]))
        i = 0
        for index, row_o, col_o, rev in self.__features_overlapping:
            for row_d, col_d in empty_pixels:
                cost_table[int(i / (empty_pixels.shape[0])), i % (empty_pixels.shape[0])] = (abs(row_d - row_o) + abs(col_d - col_o)) * rev
                i = i + 1
        return cost_table

    def __get_features_overlapping(self,features_data):
        """
        Compute the features that need to be relocated using the features original positions and the relevance of each feature. If a pixel is shared by several features,
         the mapping is maintained with the most relevant feature and the others must be relocated.
        Args:
            features_data: Array with features original positions

        Returns:
        Array with the index of features that need to be relocated and its relevance
        """
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
        """
        Create structure with empty pixels according to dimension
        Args:
            features_data: features information
            dim:image dim

        Returns:
        Array with empty pixels positions
        """
        image_matrix = np.zeros((dim, dim))
        for i, j in zip(features_data[:, 0], features_data[:, 1]):
            image_matrix[int(i), int(j)] += 1
        empty_pixels = np.argwhere(image_matrix == 0)
        return empty_pixels

    def _get_features_positions(self,features_data,empty_pixels,row_idxs,pixels_idxs):
        """
        Get features positions array according to features coordinates, empty pixels array and mapping between pixel and features optimized
        Args:
            features_data: features original positions
            empty_pixels: Array with empty pixels positions
            row_idxs: Index of features
            pixels_idxs: Index of pixels

        Returns:
        Array with image features positions
        """
        idxs=self.__features_overlapping[row_idxs,3].astype(int)
        features_data=features_data[:,:2]
        features_data[idxs] = empty_pixels[pixels_idxs]
        return features_data