# Standard library imports
import os
import shutil
import time

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr

# Typing imports
from typing import List, Optional, Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.mappingMethod import MappingMethod


###########################################################
################    IGTD    ###############################
###########################################################

class IGTD(MappingMethod):
    """
    IGTD: Transforms tabular data into compact 2D images for CNNs by assigning features to pixels 
    based on similarity. It minimizes feature-pixel ranking differences, preserving neighborhood 
    structures effectively.

    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
   transformer : CustomTransformer, optional
        Preprocessing transformations like scaling, normalization,etc.
        Default is MinMaxScaler.
        Valid: Scikit Learn transformers or custom implementation using inheritance over CustomTransformer class.
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    scale : List[int], optional
        The number of pixel rows and columns in the image. The product (rows x columns) must 
        be greater than or equal to the number of features. 
        Default is [6,6]. Valid values: list of two positive integers.
    fea_dist_method : str, optional
        Method to calculate pairwise distances between features.
        Default is 'Pearson'. Valid values: ['Pearson', 'Spearman', 'set', 'Euclidean'].
    image_dist_method : str, optional
        Method to calculate distances between pixels in the image.
        Default is 'Euclidean'. Valid values: ['Euclidean', 'Manhattan'].
    error : str, optional
        Function to evaluate differences between feature and pixel distance rankings.
        Default is 'squared'. Valid values: ['squared', 'abs'].
    max_step : int, optional
        Maximum number of iterations for the algorithm if it does not converge. 
        Default is 1000. Valid values: integer.
    val_step : int, optional
        Number of steps to check gain on the objective function for convergence. 
        Default is 50. Valid values: integer.
    switch_t : int, optional
        Threshold for error change rate to determine if switching features should occur. 
        Default is 0. Valid values: integer.
    min_gain : float, optional
        Minimum improvement in the objective function to continue optimization. 
        Default is 0.00001. Valid values: float.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    format : str, optional
        Output format using images with matplotlib with [0,255] range for pixel or using npy format.
        Default is images with format 'png'.
    cmap : str, optional
        color map to use with matplotlib.
        Default is gray
    random_seed : int, optional
        Seed for reproducibility. 
        Default is 1. Valid values: integer.    
    """
    #Default hyperparameters
    default_scale = [6,6]                       # Characteristic pixels of the final image (row x col). [row x col] must be equal or greater than the number of features.
    
    default_fea_dist_method = "Pearson"         # 'Pearson' uses Pearson correlation coefficient to evaluate similarity between features;
                                                # 'Spearman' uses Spearman correlation coefficient to evaluate similarity between features;
                                                # 'set' uses Jaccard index to evaluate similarity between features that are binary variables;
                                                # 'Euclidean' calculates pairwise euclidean distances between features.
    default_image_dist_method = "Euclidean"     # method used to calculate distance. Can be 'Euclidean' or 'Manhattan'.
    default_error = "squared"                   # a string indicating the function to evaluate the difference between feature distance ranking and pixel distance ranking. 'abs' indicates the absolute function. 'squared' indicates the square function

    default_max_step = 1000                     # the maximum steps that the algorithm should run if never converges.
    default_val_step = 50                       # number of steps for checking gain on the objective function to determine convergence
    default_switch_t = 0                        # the threshold to determine whether switch should happen
    default_min_gain = 0.00001                  # if the objective function is not improved more than 'min_gain' in 'val_step' steps, the algorithm terminates.
    
    default_zoom = 1                            # The default multiplication value to save the image
    default_random_seed = 1                     # default seed for reproducibility
    default_cmap='gray'                         # Default cmap image output
    default_format='png'                        # Default output format

    def __init__(
        self,
        problem: Optional[str] = None,
        transformer = MinMaxScaler(),
        verbose: Optional[bool] = None,
        scale: Optional[List[int]] = default_scale,
        fea_dist_method: Optional[str] = default_fea_dist_method,
        image_dist_method: Optional[str] = default_image_dist_method,
        error: Optional[str] = default_error,
        max_step: Optional[int] = default_max_step,
        val_step: Optional[int] = default_val_step,
        switch_t: Optional[int] = default_switch_t,
        min_gain: Optional[float] = default_min_gain,
        zoom: Optional[int] = default_zoom,
        format=default_format,
        cmap=default_cmap,
        random_seed: Optional[int] = default_random_seed,
    ):
        super().__init__(problem=problem, verbose=verbose, transformer=transformer, zoom=zoom,format=format, cmap=cmap)

        self.scale = scale
        self.fea_dist_method = fea_dist_method
        self.image_dist_method = image_dist_method
        self.zoom = zoom
        self.max_step = max_step
        self.val_step = val_step
        self.error = error
        self.switch_t = switch_t
        self.min_gain = min_gain
        self.random_seed = random_seed

    def __min_max_transform(self, data: np.ndarray):
        '''
        This function does a linear transformation of each feature, so that the minimum and maximum values of a
        feature are 0 and 1, respectively.

        Input
        -----
        data: ndarray
            an input data array with a size of [n_sample, n_feature]

        Return
        ------
        norm_data: ndarray
            the data array after transformation
        '''

        norm_data = np.empty(data.shape)
        norm_data.fill(np.nan)
        for i in range(data.shape[1]):
            v = data[:, i].copy()
            if np.max(v) == np.min(v):
                norm_data[:, i] = 0
            else:
                v = (v - np.min(v)) / (np.max(v) - np.min(v))
                norm_data[:, i] = v
        return norm_data

    def __generate_feature_distance_ranking(self, data:np.ndarray):
        '''
        This function generates ranking of distances/dissimilarities between features for tabular data.

        Input
        -----
        data: np.ndarray
            input data, n_sample by n_feature

        Return
        ------
        ranking: nd.ndarray
            symmetric ranking matrix based on dissimilarity
        corr:
            matrix of distances between features
        '''

        num = data.shape[1]
        if self.fea_dist_method == 'Pearson':
            corr = np.corrcoef(np.transpose(data))
        elif self.fea_dist_method == 'Spearman':
            corr = spearmanr(data).correlation
        elif self.fea_dist_method == 'Euclidean':
            corr = squareform(pdist(np.transpose(data), metric='euclidean'))
            corr = np.max(corr) - corr
            corr = corr / np.max(corr)
        elif self.fea_dist_method == 'set':  # This is the new set operation to calculate similarity. It does not tolerate all-zero features.
            corr1 = np.dot(np.transpose(data), data)
            corr2 = data.shape[0] - np.dot(np.transpose(1 - data), 1 - data)
            corr = corr1 / corr2

        corr = 1 - corr
        corr = np.around(a=corr, decimals=10)

        tril_id = np.tril_indices(num, k=-1)
        rank = rankdata(corr[tril_id])
        ranking = np.zeros((num, num))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)

        return ranking, corr

    def __generate_matrix_distance_ranking(self, num_r: int, num_c: int, num: int):
        '''
        This function calculates the ranking of distances between all pairs of entries in a matrix of size num_r by num_c.

        Input
        -----
        num_r: int
            number of rows in the matrix
        num_c: int
            number of columns in the matrix
        num: int

        Return
        ------
        coordinate: np.ndarray
            num_r * num_c by 2 matrix giving the coordinates of elements in the matrix.
        ranking: np.ndarray
            a num_r * num_c by num_r * num_c matrix giving the ranking of pair-wise distance.
        '''

        # generate the coordinates of elements in a matrix
        for r in range(num_r):
            if r == 0:
                coordinate = np.transpose(np.vstack((np.zeros(num_c), range(num_c))))
            else:
                coordinate = np.vstack((coordinate, np.transpose(np.vstack((np.ones(num_c) * r, range(num_c))))))
        coordinate = coordinate[:num, :]

        # calculate the closeness of the elements
        cord_dist = np.zeros((num, num))
        if self.image_dist_method == 'Euclidean':
            for i in range(num):
                cord_dist[i, :] = np.sqrt(np.square(coordinate[i, 0] * np.ones(num) - coordinate[:, 0]) +
                                          np.square(coordinate[i, 1] * np.ones(num) - coordinate[:, 1]))
        elif self.image_dist_method == 'Manhattan':
            for i in range(num):
                cord_dist[i, :] = np.abs(coordinate[i, 0] * np.ones(num) - coordinate[:, 0]) + \
                                  np.abs(coordinate[i, 1] * np.ones(num) - coordinate[:, 1])

        # generate the ranking based on distance
        tril_id = np.tril_indices(num, k=-1)
        rank = rankdata(cord_dist[tril_id])
        ranking = np.zeros((num, num))
        ranking[tril_id] = rank
        ranking = ranking + np.transpose(ranking)

        coordinate = np.int64(coordinate)
        return (coordinate[:, 0], coordinate[:, 1]), ranking

    def __IGTD_absolute_error(self, source, target):
        '''
        This function switches the order of rows (columns) in the source ranking matrix to make it similar to the target
        ranking matrix. In each step, the algorithm randomly picks a row that has not been switched with others for
        the longest time and checks all possible switch of this row, and selects the switch that reduces the
        dissimilarity most. Dissimilarity (i.e. the error) is the summation of absolute difference of
        lower triangular elements between the rearranged source ranking matrix and the target ranking matrix.

        Input
        -----
        source:
            a symmetric ranking matrix with zero diagonal elements.
        target:
            a symmetric ranking matrix with zero diagonal elements with the same size as 'source'.

        Return
        ------
        index_record:
            indices to rearrange the rows(columns) in source obtained the optimization process
        err_record:
            error obtained in the optimization process
        run_time:
            the time at which each step is completed in the optimization process
        '''

        r = np.random.RandomState(seed=self.random_seed)

        source = source.copy()
        num = source.shape[0]
        tril_id = np.tril_indices(num, k=-1)
        index = np.array(range(num))
        index_record = np.empty((self.max_step + 1, num))
        index_record.fill(np.nan)
        index_record[0, :] = index.copy()

        # calculate the error associated with each row
        err_v = np.empty(num)
        err_v.fill(np.nan)
        for i in range(num):
            err_v[i] = np.sum(np.abs(source[i, 0:i] - target[i, 0:i])) + \
                       np.sum(np.abs(source[(i + 1):, i] - target[(i + 1):, i]))

        step_record = -np.ones(num)
        err_record = [np.sum(abs(source[tril_id] - target[tril_id]))]
        pre_err = err_record[0]
        t1 = time.time()
        run_time = [0]

        for s in range(self.max_step):
            delta = np.ones(num) * np.inf

            # randomly pick a row that has not been considered for the longest time
            idr = np.where(step_record == np.min(step_record))[0]
            ii = idr[r.permutation(len(idr))[0]]

            for jj in range(num):
                if jj == ii:
                    continue

                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii

                err_ori = err_v[i] + err_v[j] - np.abs(source[j, i] - target[j, i])

                err_i = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                        np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(source[i, j] - target[j, i])
                err_j = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                        np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(source[i, j] - target[j, i])
                err_test = err_i + err_j - np.abs(source[i, j] - target[j, i])

                delta[jj] = err_test - err_ori

            delta_norm = delta / pre_err
            id = np.where(delta_norm <= self.switch_t)[0]
            if len(id) > 0:
                jj = np.argmin(delta)

                # Update the error associated with each row
                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii
                for k in range(num):
                    if k < i:
                        err_v[k] = err_v[k] - np.abs(source[i, k] - target[i, k]) - np.abs(source[j, k] - target[j, k]) + \
                                   np.abs(source[j, k] - target[i, k]) + np.abs(source[i, k] - target[j, k])
                    elif k == i:
                        err_v[k] = np.sum(np.abs(source[j, :i] - target[i, :i])) + \
                                   np.sum(np.abs(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                                   np.sum(np.abs(source[(j + 1):, j] - target[(j + 1):, i])) + np.abs(
                            source[i, j] - target[j, i])
                    elif k < j:
                        err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[j, k] - target[j, k]) + \
                                   np.abs(source[k, j] - target[k, i]) + np.abs(source[i, k] - target[j, k])
                    elif k == j:
                        err_v[k] = np.sum(np.abs(source[i, :i] - target[j, :i])) + \
                                   np.sum(np.abs(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                                   np.sum(np.abs(source[(j + 1):, i] - target[(j + 1):, j])) + np.abs(
                            source[i, j] - target[j, i])
                    else:
                        err_v[k] = err_v[k] - np.abs(source[k, i] - target[k, i]) - np.abs(source[k, j] - target[k, j]) + \
                                   np.abs(source[k, j] - target[k, i]) + np.abs(source[k, i] - target[k, j])

                # switch rows i and j
                ii_v = source[ii, :].copy()
                jj_v = source[jj, :].copy()
                source[ii, :] = jj_v
                source[jj, :] = ii_v
                ii_v = source[:, ii].copy()
                jj_v = source[:, jj].copy()
                source[:, ii] = jj_v
                source[:, jj] = ii_v
                err = delta[jj] + pre_err

                # update rearrange index
                t = index[ii]
                index[ii] = index[jj]
                index[jj] = t

                # update step record
                step_record[ii] = s
                step_record[jj] = s
            else:
                # error is not changed due to no switch
                err = pre_err

                # update step record
                step_record[ii] = s

            err_record.append(err)
            index_record[s + 1, :] = index.copy()
            run_time.append(time.time() - t1)

            if s > self.val_step:
                if np.sum((err_record[-self.val_step - 1] - np.array(err_record[(-self.val_step):])) / err_record[
                    -self.val_step - 1] >= self.min_gain) == 0:
                    break

            pre_err = err
        index_record = index_record[:len(err_record), :].astype(int)
        return index_record, err_record, run_time

    def __IGTD_square_error(self, source, target):
        '''
        This function switches the order of rows (columns) in the source ranking matrix to make it similar to the target
        ranking matrix. In each step, the algorithm randomly picks a row that has not been switched with others for
        the longest time and checks all possible switch of this row, and selects the switch that reduces the
        dissimilarity most. Dissimilarity (i.e. the error) is the summation of squared difference of
        lower triangular elements between the rearranged source ranking matrix and the target ranking matrix.

        Input
        -----
        source:
            a symmetric ranking matrix with zero diagonal elements.
        target:
            a symmetric ranking matrix with zero diagonal elements. 'source' and 'target' should have the same size.

        Return
        ------
        index_record:
            ordering index to rearrange the rows (columns) in 'source' in the optimization process
        err_record:
            the error history in the optimization process
        run_time:
            the time at which each step is finished in the optimization process
        '''

        r = np.random.RandomState(seed=self.random_seed)

        source = source.copy()
        num = source.shape[0]
        tril_id = np.tril_indices(num, k=-1)
        index = np.array(range(num))
        index_record = np.empty((self.max_step + 1, num))
        index_record.fill(np.nan)
        index_record[0, :] = index.copy()

        # calculate the error associated with each row
        err_v = np.empty(num)
        err_v.fill(np.nan)
        for i in range(num):
            err_v[i] = np.sum(np.square(source[i, 0:i] - target[i, 0:i])) + \
                       np.sum(np.square(source[(i + 1):, i] - target[(i + 1):, i]))

        step_record = -np.ones(num)
        err_record = [np.sum(np.square(source[tril_id] - target[tril_id]))]
        pre_err = err_record[0]
        t1 = time.time()
        run_time = [0]

        for s in range(self.max_step):
            delta = np.ones(num) * np.inf

            # randomly pick a row that has not been considered for the longest time
            idr = np.where(step_record == np.min(step_record))[0]
            ii = idr[r.permutation(len(idr))[0]]

            for jj in range(num):
                if jj == ii:
                    continue

                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii

                err_ori = err_v[i] + err_v[j] - np.square(source[j, i] - target[j, i])

                err_i = np.sum(np.square(source[j, :i] - target[i, :i])) + \
                        np.sum(np.square(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                        np.sum(np.square(source[(j + 1):, j] - target[(j + 1):, i])) + np.square(
                    source[i, j] - target[j, i])
                err_j = np.sum(np.square(source[i, :i] - target[j, :i])) + \
                        np.sum(np.square(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                        np.sum(np.square(source[(j + 1):, i] - target[(j + 1):, j])) + np.square(
                    source[i, j] - target[j, i])
                err_test = err_i + err_j - np.square(source[i, j] - target[j, i])

                delta[jj] = err_test - err_ori

            delta_norm = delta / pre_err
            id = np.where(delta_norm <= self.switch_t)[0]
            if len(id) > 0:
                jj = np.argmin(delta)

                # Update the error associated with each row
                if ii < jj:
                    i = ii
                    j = jj
                else:
                    i = jj
                    j = ii
                for k in range(num):
                    if k < i:
                        err_v[k] = err_v[k] - np.square(source[i, k] - target[i, k]) - np.square(
                            source[j, k] - target[j, k]) + \
                                   np.square(source[j, k] - target[i, k]) + np.square(source[i, k] - target[j, k])
                    elif k == i:
                        err_v[k] = np.sum(np.square(source[j, :i] - target[i, :i])) + \
                                   np.sum(np.square(source[(i + 1):j, j] - target[(i + 1):j, i])) + \
                                   np.sum(np.square(source[(j + 1):, j] - target[(j + 1):, i])) + np.square(
                            source[i, j] - target[j, i])
                    elif k < j:
                        err_v[k] = err_v[k] - np.square(source[k, i] - target[k, i]) - np.square(
                            source[j, k] - target[j, k]) + \
                                   np.square(source[k, j] - target[k, i]) + np.square(source[i, k] - target[j, k])
                    elif k == j:
                        err_v[k] = np.sum(np.square(source[i, :i] - target[j, :i])) + \
                                   np.sum(np.square(source[i, (i + 1):j] - target[j, (i + 1):j])) + \
                                   np.sum(np.square(source[(j + 1):, i] - target[(j + 1):, j])) + np.square(
                            source[i, j] - target[j, i])
                    else:
                        err_v[k] = err_v[k] - np.square(source[k, i] - target[k, i]) - np.square(
                            source[k, j] - target[k, j]) + \
                                   np.square(source[k, j] - target[k, i]) + np.square(source[k, i] - target[k, j])

                # switch rows i and j
                ii_v = source[ii, :].copy()
                jj_v = source[jj, :].copy()
                source[ii, :] = jj_v
                source[jj, :] = ii_v
                ii_v = source[:, ii].copy()
                jj_v = source[:, jj].copy()
                source[:, ii] = jj_v
                source[:, jj] = ii_v
                err = delta[jj] + pre_err

                # update rearrange index
                t = index[ii]
                index[ii] = index[jj]
                index[jj] = t

                # update step record
                step_record[ii] = s
                step_record[jj] = s
            else:
                # error is not changed due to no switch
                err = pre_err

                # update step record
                step_record[ii] = s

            err_record.append(err)
            index_record[s + 1, :] = index.copy()
            run_time.append(time.time() - t1)

            if s > self.val_step:
                if np.sum((err_record[-self.val_step - 1] - np.array(err_record[(-self.val_step):])) / err_record[
                    -self.val_step - 1] >= self.min_gain) == 0:
                    break

            pre_err = err

        index_record = index_record[:len(err_record), :].astype(int)

        return index_record, err_record, run_time

    def __training(self, source, target):
        '''
        This is just a wrapper function that wraps the two search functions using different error measures.
        '''
        if self.error  == 'abs':
            index_record, err_record, run_time = self.__IGTD_absolute_error(source=source, target=target)
        if self.error == 'squared':
            index_record, err_record, run_time = self.__IGTD_square_error(source=source, target=target)

        return index_record, err_record, run_time

    def __generate_image_data(self, data, index, num_row, num_column, coord, labels):
        '''
        This function generates the data in image format according to rearrangement indices. It saves the data
        sample-by-sample in both txt files and image files.

        Input
        -----
        data:
            original tabular data, 2D array or data frame, n_samples by n_features.
        index:
            indices of features obtained through optimization, according to which the features can be
            arranged into a num_r by num_c image.
        num_row: int
            number of rows in image.
        num_column: int
            number of columns in image.
        coord:
            coordinates of features in the image/matrix.
        labels: ndarray
            ndarray containing the labels for the tabular data

        Return
        ------
        image_data:
            the generated data, a 3D numpy array. The third dimension is across samples. The range of values
            is [0, 255]. Small values actually indicate high values in the original data.
        samples:
            the names of indices of the samples.
        '''

        if isinstance(data, pd.DataFrame):
            samples = data.index.map(np.str)
            data = data.values
        else:
            samples = [str(i) for i in range(data.shape[0])]

        data_2 = data.copy()
        data_2 = data_2[:, index]
        max_v = np.max(data_2)
        min_v = np.min(data_2)
        data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255  # So that black means high value

        image_data = np.empty((num_row, num_column, data_2.shape[0]))
        image_data.fill(np.nan)


        for i in range(data_2.shape[0]):
            data_i = np.empty((num_row, num_column))
            data_i.fill(np.nan)
            data_i[coord] = data_2[i, :]

            iid = np.where(np.isnan(data_i))
            data_i[iid] = 255

            image_data[:, :, i] = data_i
            image_data[:, :, i] = 255 - image_data[:, :, i]

            self._save_image(data_i,labels[i],i)

        return image_data, samples

    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        # Check if the dimensions are correct ( Attributes => Scale[n,m].size )
        numPixels = self.scale[0] * self.scale[1]
        numAttributes = x.shape[1]
        if numAttributes > numPixels:
            Exception(f"Error: Attributes can't be wrapped in {self.scale}, scale. Please user higher scale.")

        X = x.values
        Y = y.values if y is not None else None

        ranking_feature, corr = self.__generate_feature_distance_ranking(data=X)
        self.coordinate, ranking_image = self.__generate_matrix_distance_ranking(num_r=self.scale[0], num_c=self.scale[1], num=X.shape[1])
        self.index, err, time = self.__training(source=ranking_feature, target=ranking_image)

        self.min_id = np.argmin(err)
        self._build_features_mapping(x.columns,np.column_stack((self.index[self.min_id, :]//self.scale[1],self.index[self.min_id, :]%self.scale[1])))

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        # Check if the dimensions are correct ( Attributes => Scale[n,m].size )
        numPixels = self.scale[0] * self.scale[1]
        numAttributes = x.shape[1]
        if numAttributes > numPixels:
            Exception(f"Error: Attributes can't be wrapped in {self.scale}, scale. Please user higher scale.")

        X = x.values 
        Y = y.values if y is not None else None

        self.__generate_image_data(
            data=X,
            index=self.index[self.min_id, :],
            num_row=self.scale[0],
            num_column=self.scale[1],
            coord=self.coordinate,
            labels=Y
        )
        self._features_mapping_to_csv()
