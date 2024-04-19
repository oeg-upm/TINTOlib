from scipy.stats import spearmanr, rankdata
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
import time
import pickle
from typing import List

class IGTD:
    #Default hyperparameters
    default_scale = [6,6]
    default_fea_dist_method = "Pearson" # 'Pearson' uses Pearson correlation coefficient to evaluate similarity between features;
                                        # 'Spearman' uses Spearman correlation coefficient to evaluate similarity between features;
                                        # 'set' uses Jaccard index to evaluate similarity between features that are binary variables;
                                        # 'Euclidean' calculates pairwise euclidean distances between features.
    default_image_dist_method = "Euclidean" # method used to calculate distance. Can be 'Euclidean' or 'Manhattan'.
    default_save_image_size = 20            # Size in inches to save the image
    default_max_step = 1000                 # the maximum steps that the algorithm should run if never converges.
    default_val_step = 50                   # number of steps for checking gain on the objective function to determine convergence
    default_error = "squared"               # a string indicating the function to evaluate the difference between feature distance ranking and pixel distance ranking. 'abs' indicates the absolute function. 'squared' indicates the square function.
    default_switch_t = 0                    # the threshold to determine whether switch should happen
    default_min_gain = 0.00001              # if the objective function is not improved more than 'min_gain' in 'val_step' steps, the algorithm terminates.
    default_seed = 1                        # for setting random seed.
    default_verbose = False
    default_problem = "supervised"          # Define the type of dataset [supervised, unsupervised, regression]
    
    def __init__(self,
                 problem: str = default_problem,
                 scale: List[int] = default_scale,
                 fea_dist_method: str = default_fea_dist_method,
                 image_dist_method: str = default_image_dist_method,
                 save_image_size: int = default_save_image_size,
                 max_step: int = default_max_step,
                 val_step: int = default_val_step,
                 error: str = default_error,
                 switch_t: int =default_switch_t,
                 min_gain: float =default_min_gain,
                 seed = default_seed,
                 verbose: bool = default_verbose
    ):
        '''
        Input
        -----
        problem:
        scale: List[int]
            a list of two positive integers. The number of pixel rows and columns in the image representations,
            into which the tabular data will be converted.
        fea_dist_method: str
            a string indicating the method used for calculating the pairwise distances between features,
            for which there are three options.
            'Pearson' uses the Pearson correlation coefficient to evaluate the similarity between features.
            'Spearman' uses the Spearman correlation coefficient to evaluate the similarity between features.
            'Euclidean' calculates pairwise euclidean distances between features.
            'set' uses the Jaccard index to evaluate the similarity between features that are binary variables.
        image_dist_method: str
            a string indicating the method used for calculating the distances between pixels in image.
            It can be either 'Euclidean' or 'Manhattan'.
        save_image_size: (optional) int
            Defaults to None. The size in inches for saving the visual results. If save_image_size is None,
            the resulting images will have a size of scale[0]*scale[1] pixels. Otherwise, save_image_size represents
            the size in inches of the resulting images.
        max_step: int
            the maximum number of iterations that the IGTD algorithm will run if never converges.
        val_step: int
            the number of iterations for determining algorithm convergence. If the error reduction rate is smaller than
            min_gain for val_step iterations, the algorithm converges.
        error: str
            name of the function to evaluate the difference between feature distance ranking and pixel
            distance ranking. 'abs' indicates the absolute function. 'squared' indicates the square function.
        switch_t: 
            the threshold on error change rate. Error change rate is
            (error after feature swapping - error before feature swapping) / error before feature swapping.
            In each iteration, if the smallest error change rate resulted from all possible feature swappings
            is not larger than switch_t, the feature swapping resulting in the smallest error change rate will
            be performed. If switch_t <= 0, the IGTD algorithm monotonically reduces the error during optimization.
        min_gain: 
            if the error reduction rate is not larger than min_gain for val_step iterations, the algorithm converges.
        seed: int
            seed to make results reproducible
        verbose: bool
            whether to print progress on the terminal
        '''
        self.scale: List[int] = scale
        self.fea_dist_method: str = fea_dist_method
        self.image_dist_method: str = image_dist_method
        self.save_image_size: int = save_image_size
        self.max_step: int = max_step
        self.val_step: int = val_step
        self.error: str = error
        self.switch_t: int = switch_t
        self.min_gain: float = min_gain
        self.seed: int = seed
        self.verbose: bool = verbose
        self.problem: str = problem

    def saveHyperparameters(self, filename='objs'):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.

        Input
        -----
        filename: str
            Name for the pickle file.
        """
        with open(filename + ".pkl", 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print("It has been successfully saved in " + filename)

    def loadHyperparameters(self, filename='objs.pkl'):
        """
        This function allows LOADING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.

        Input
        -----
        filename: str
            Name of the pickle file.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
            self.scale = variables["scale"]
            self.fea_dist_method = variables["fea_dist_method"]
            self.image_dist_method = variables["image_dist_method"]
            self.save_image_size = variables["save_image_size"]
            self.max_step = variables["max_step"]
            self.val_step = variables["val_step"]
            self.error = variables["error"]
            self.switch_t = variables["switch_t"]
            self.min_gain = variables["min_gain"]
            self.seed = variables["seed"]
            self.verbose = variables["verbose"]

        if self.verbose:
            print("It has been successfully loaded from " + filename)

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

        r = np.random.RandomState(seed=self.seed)
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.mkdir(self.folder)

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
            if self.verbose:
                print('Step ' + str(s) + ' err: ' + str(err))
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

        r = np.random.RandomState(seed=self.seed)
        if os.path.exists(self.folder):
            shutil.rmtree(self.folder)
        os.mkdir(self.folder)

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
            if self.verbose:
                print('Step ' + str(s) + ' err: ' + str(err))
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

    def __saveSupervised(self, y, i, data_i):
        '''
        Saves the matrix as an image in a supervised dataset.

        Input
        -----
        i: int
            the index of the row within the dataset
        y:
            the true label of the i-th row
        data_i: ndarray
            the matrix containing the data to be saved as an image
        '''
        extension = 'png'  # eps o pdf
        subfolder = str(int(y)).zfill(2)  # subfolder for grouping the results of each class
        name_image = str(i).zfill(6)
        route = os.path.join(self.folder, subfolder)
        route_complete = os.path.join(route, name_image + '.' + extension)
        # Subfolder check
        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")

        if self.save_image_size is None:
            plt.imsave(route_complete, data_i, cmap='gray', vmin=0, vmax=255)
        else:
            fig = plt.figure(figsize=(self.save_image_size, self.save_image_size))
            plt.imshow(data_i, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.savefig(fname=route_complete, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __saveRegressionOrUnsupervised(self, i, data_i):
        '''
        Saves the matrix as an image in a regression or unsupervised dataset.

        Input
        -----
        i: int
            the index of the row within the dataset
        data_i: ndarray
            the matrix containing the data to be saved as an image
        '''
        extension = 'png'  # eps o pdf
        subfolder = "images"
        name_image = str(i).zfill(6) + '.' + extension
        route = os.path.join(self.folder, subfolder)
        route_complete = os.path.join(route, name_image)
        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")

        if self.save_image_size is None:
            plt.imsave(route_complete, data_i, cmap='gray', vmin=0, vmax=255)
        else:
            fig = plt.figure(figsize=(self.save_image_size, self.save_image_size))
            plt.imshow(data_i, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.savefig(fname=route_complete, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        route_relative = os.path.join(subfolder, name_image)
        return route_relative
    
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
        image_folder = self.folder
        imagesRoutesArr = []
        if isinstance(data, pd.DataFrame):
            samples = data.index.map(np.str)
            data = data.values
        else:
            samples = [str(i) for i in range(data.shape[0])]

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)
        os.mkdir(image_folder)

        data_2 = data.copy()
        data_2 = data_2[:, index]
        max_v = np.max(data_2)
        min_v = np.min(data_2)
        data_2 = 255 - (data_2 - min_v) / (max_v - min_v) * 255  # So that black means high value

        image_data = np.empty((num_row, num_column, data_2.shape[0]))
        image_data.fill(np.nan)
        total = data_2.shape[0]
        for i in range(data_2.shape[0]):
            data_i = np.empty((num_row, num_column))
            data_i.fill(np.nan)
            data_i[coord] = data_2[i, :]
            
            iid = np.where(np.isnan(data_i))
            data_i[iid] = 255

            image_data[:, :, i] = data_i
            image_data[:, :, i] = 255 - image_data[:, :, i]
            if image_folder is not None:
                if self.problem == "supervised":
                    route = self.__saveSupervised(labels[i], i, data_i)
                    imagesRoutesArr.append(route)
                elif self.problem == "unsupervised" or self.problem == "regression":
                    route = self.__saveRegressionOrUnsupervised(i, data_i)
                    imagesRoutesArr.append(route)
                else:
                    print("Wrong problem definition. Please use 'supervised', 'unsupervised' or 'regression'")

                # Verbose
                if self.verbose:
                    print("Created ", str(i + 1), "/", int(total))

        if self.problem == "supervised":
            data = {'images': imagesRoutesArr, 'class': labels}
            supervisedCSV = pd.DataFrame(data=data)
            supervisedCSV.to_csv(self.folder + "/supervised.csv", index=False)
        elif self.problem == "unsupervised":
            data = {'images': imagesRoutesArr}
            unsupervisedCSV = pd.DataFrame(data=data)
            unsupervisedCSV.to_csv(self.folder + "/unsupervised.csv", index=False)
        elif self.problem == "regression":
            data = {'images': imagesRoutesArr, 'values': labels}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/regression.csv", index=False)

        return image_data, samples

    def __trainingAlg(self, X: np.ndarray, Y: np.ndarray):
        '''
        Function that calls all the auxiliary functions from start to end to calculate and save the images.

        Input
        -----
        X: ndarray
            ndarray containing the tabular data
        Y: ndarray
            ndarray containing the labels for the tabular data
        '''
        ranking_feature, corr = self.__generate_feature_distance_ranking(data=X)
        coordinate, ranking_image = self.__generate_matrix_distance_ranking(num_r=self.scale[0], num_c=self.scale[1], num=X.shape[1])
        index, err, time = self.__training(source=ranking_feature, target=ranking_image)

        min_id = np.argmin(err)
        #ranking_feature_random = ranking_feature[index[min_id, :], :]
        #ranking_feature_random = ranking_feature_random[:, index[min_id, :]]

        X, samples = self.__generate_image_data(
            data=X,
            index=index[min_id, :],
            num_row=self.scale[0],
            num_column=self.scale[1],
            coord=coordinate,
            labels=Y
        )
        
        if self.verbose:
            print("End")

    def generateImages(self, data, folder="/igtd_files"):
        '''
        This function converts tabular data into images using the IGTD algorithm.
        This function does not return any variable, but saves multiple result files, which are the following
        1.  Results.pkl stores the original tabular data, the generated image data, and the names of samples. The generated
            image data is a 3D numpy array. Its size is [number of pixel rows in image, number of pixel columns in image,
            number of samples]. The range of values is [0, 255]. Small values in the array actually correspond to high
            values in the tabular data.
        2.  Results_Auxiliary.pkl stores the ranking matrix of pairwise feature distances before optimization,
            the ranking matrix of pairwise pixel distances, the coordinates of pixels when concatenating pixels
            row by row from image to form the pixel distance ranking matrix, error in each iteration,
            and time (in seconds) when completing each iteration.
        3.  original_feature_ranking.png shows the feature distance ranking matrix before optimization.
        4.  image_ranking.png shows the pixel distance ranking matrix.
        5.  error_and_runtime.png shows the change of error vs. time during the optimization process.
        6.  error_and_iteration.png shows the change of error vs. iteration during the optimization process.
        7.  optimized_feature_ranking.png shows the feature distance ranking matrix after optimization.
        8.  data folder includes two image data files for each sample. The txt file is the image data in matrix format.
            The png file shows the visualization of image data.

        Input
        -----
        data: str or pd.DataFrame
            The path to a csv, or a data frame contaning the tabular data. Its size is n_samples by n_features.
        folder: str
            The path to save the results
        '''

        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)

        # Read the CSV
        self.folder = folder
        if type(data) == str:
            dataset = pd.read_csv(data)
            array = dataset.values
        elif isinstance(data, pd.DataFrame):
            array = data.values

        # Separate X from Y
        X = self.__min_max_transform(array[:, :-1])     # Remove the last column
        Y = array[:, -1]

        # Check if the dimensions are correct ( Attributes => Scale[n,m].size )
        numPixels=self.scale[0]*self.scale[1]
        numAttributes = X.shape[1]
        if numAttributes > numPixels:
            error_text = f"Error: Attributes can't be wrapped in {self.scale}, scale. Please user higher scale."
            raise Exception(error_text)

        self.__trainingAlg(X, Y)
