import numpy as np
import pandas as pd
import os
import gc

# Dimensional reduction classes
from sklearn.manifold import TSNE
# from tsnecuda import TSNE
from sklearn.decomposition import PCA

# Sklearn
from sklearn.preprocessing import MinMaxScaler

# Graphic library
import matplotlib
import matplotlib.image

# Additional libraries
import math
import pickle

###########################################################
################    TINTO EXECUTION    ####################
###########################################################

###### default values ###############

default_algorithm = "PCA"  # Dimensionality reduction algorithm (PCA o t-SNE)
default_pixeles = 20  # Image's Pixels (one side)

default_blur = False  # Active option blurring
default_amplification = np.pi  # Amplification in blurring
default_distance = 2  # Distance in blurring (number of pixels)
default_steps = 4  # Steps in blurring
default_option = 'mean'  # Option in blurring (mean and maximum)

default_seed = 20  # Seed
default_times = 4  # Times replication in t-SNE
default_verbose = False  # Verbose: if it's true, show the compilation text

# A parte funciones
default_save = False  # Save configurations (to reuse)
default_load = False  # Load configurations (.pkl)


class tinto:
    def __init__(self, src_data=None, dest_folder=None, algorithm=default_algorithm, pixels=default_pixeles, blur=default_blur,
                 amplification=default_amplification, distance=default_distance, steps=default_steps, option=default_option,
                 seed=default_seed, times=default_times, verbose=default_verbose):

        self.algorithm = algorithm
        self.pixels = pixels

        self.blur = blur
        self.amplification = amplification
        self.distance = distance
        self.steps = steps
        self.option = option

        self.seed = seed
        self.times = times
        self.verbose = verbose

        self.src_data = src_data  # Source location (tidy data in csv without head)
        self.dest_folder = dest_folder  # Destination location (folder)

        self.error_pos = False  # Indicates the overlap of characteristic pixels.

    def saveHyperparameters(self, filename='objs'):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self.__dict__, f)
        if (self.verbose):
            print("It has been successfully saved in " + filename)

    def loadHyperparameters(self, filename='objs.pkl'):
        """
        This function allows LOADING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
            self.algorithm = variables["algorithm"]
            self.pixels = variables["pixels"]

            self.blur = variables["blur"]
            self.amplification = variables["amplification"]
            self.distance = variables["distance"]
            self.steps = variables["steps"]
            self.option = variables["option"]

            self.seed = variables["seed"]
            self.times = variables["times"]
            self.verbose = variables["verbose"]

            self.src_data = variables["src_data"]  # Source location (tidy data in csv without head)
            self.dest_folder = variables["dest_folder"]  # Destination location (folder)

        if self.verbose:
            print("It has been successfully loaded in " + filename)


    def __square(self, coord):
        """
        This functionhas the purpose of being able to create the square delimitation of the resulting image.
        The steps it performs in the order of the code are as follows:
            - Calculate the average of the points $(x,y)$.
            - Centres the data at the point $(0,0)$.
            - Calculate the distance from the centre to the points.
            - The larger distance of \texttt{dista}, transforms it to integer.
            - Calculate the vertices of the square.
            - Move the points to quadrant $4$.
            - Transfers the vertices as well.
            - Returns the values, coordinates, and vertices.
        """
        m = np.mean(coord, axis=0).reshape((1, 2))
        coord_new = coord - m
        dista = (coord_new[:, 0] ** 2 + coord_new[:, 1] ** 2) ** 0.5
        maxi = math.ceil(max(dista))
        vertices = np.array([[-maxi, maxi], [-maxi, -maxi], [maxi, -maxi], [maxi, maxi]])
        coord_new = coord_new - vertices[0]
        vertices = vertices - vertices[0]
        return coord_new, vertices

    def __m_imagen(self, coord, vertices):
        """
        This function obtain the coordinates of the matrix. This function has
        the following specifications:
            - Create a matrix of coordinates and vertices.
            - Transform the coordinates into indices for the matrix.
            - Fill in the positions of the features.
            - Finally, a conditional is created if the features were grouped
              in the same position.
        """
        pixels = self.pixels
        size = (pixels, pixels)
        matrix = np.zeros(size)

        coord_m = (coord / vertices[2, 0]) * (pixels - 1)
        coord_m = np.round(abs(coord_m))

        for i, j in zip(coord_m[:, 1], coord_m[:, 0]):
            matrix[int(i), int(j)] = 1

        if (np.count_nonzero(matrix != 0) != coord.shape[0]):
            return coord_m, matrix, True
        else:
            return coord_m, matrix, False

    def __createFilter(self):
        """
        In this function a filter is created since a matrix of size "2*distance*total_steps+1"
        is being created to act as a "filter", which covers the whole circular space of the minutiae
        determined by the distance and by the total number of steps.
        This "filter", which is a matrix, would be multiplied with a scalar, which is the intensity value.
        Finally, this resulting matrix is placed as a submatrix within the final matrix where the centre
        of the submatrix would be the position of the characteristic pixel.
        """
        distance = self.distance
        steps = self.steps
        amplification = self.amplification
        size_filter = int(2 * distance * steps + 1)
        center_x = distance * steps
        center_y = distance * steps
        print(distance, steps)
        filter = np.zeros([size_filter, size_filter])

        for step in reversed(range(steps)):
            r_actual = int(distance * (step + 1))  # current radius from largest to smallest

            # Function of intensity
            intensity = min(amplification * 1 / (np.pi * r_actual ** 2), 1)

            # Delimitation of the area
            lim_inf_i = max(center_x - r_actual - 1, 0)
            lim_sup_i = min(center_x + r_actual + 1, size_filter)
            lim_inf_j = max(center_y - r_actual - 1, 0)
            lim_sup_j = min(center_y + r_actual + 1, size_filter)

            # Allocation of values
            for i in range(lim_inf_i, lim_sup_i):
                for j in range(lim_inf_j, lim_sup_j):
                    if ((center_x - i) ** 2 + (center_y - j) ** 2 <= r_actual ** 2):
                        filter[i, j] = intensity
        filter[center_x, center_y] = 1
        return filter

    def __blurringFilter(self, matrix, filter, values, coordinates):
        """
       This function is to be able to add more ordered contextual information to the image through the
       classical painting technique called blurring. This function develops the following main steps:
       - Take the coordinate matrix of the characteristic pixels.
       - Create the blurring according to the number of steps taken in a loop with the
         following specifications:
            - Delimit the blurring area according to $(x,y)$ on an upper and lower boundary.
            - Set the new intensity values in the matrix, taking into account that if there is
              pixel overlap, the maximum or average will be taken as specified.
        """
        option = self.option

        iter_values = iter(values)
        size_matrix = matrix.shape[0]
        size_filter = filter.shape[0]
        matrix_extended = np.zeros([size_filter + size_matrix, size_filter + size_matrix])
        matrix_add = np.zeros([size_filter + size_matrix, size_filter + size_matrix])
        center_filter = int((size_filter - 1) / 2)
        for i, j in coordinates:
            i = int(i)
            j = int(j)
            value = next(iter_values)
            submatrix = filter * value

            # Delimitación del área
            lim_inf_i = i
            lim_sup_i = i + 2 * center_filter + 1
            lim_inf_j = j
            lim_sup_j = j + 2 * center_filter + 1

            if option == 'mean':
                matrix_extended[lim_inf_i:lim_sup_i, lim_inf_j:lim_sup_j] += submatrix
                matrix_add[lim_inf_i:lim_sup_i, lim_inf_j:lim_sup_j] += (submatrix > 0) * 1
            elif option == 'maximum':
                matrix_extended[lim_inf_i:lim_sup_i, lim_inf_j:lim_sup_j] = np.maximum(
                    matrix_extended[lim_inf_i:lim_sup_i, lim_inf_j:lim_sup_j], submatrix)

        if option == 'mean':
            matrix_add[matrix_add == 0] = 1
            matrix_extended = matrix_extended / matrix_add

        matrix_final = matrix_extended[center_filter:-center_filter - 1, center_filter:-center_filter - 1]

        return matrix_final

    def __imageSampleFilter(self, X, Y, coord, matrix, folder):
        """
        This function creates the samples, i.e., the images. This function has the following specifications:
        - The first conditional performs the pre-processing of the images by creating the matrices.
        - Then the for loop generates the images for each sample. Some assumptions have to be taken into
          account in this step:
            - The samples will be created according to the number of targets. Therefore, each folder that is
              created will contain the images created for each target.
            - In the code, the images are exported in PNG format; this can be changed to any other format.
        """
        # Hyperparams
        amplification = self.amplification
        distance = self.distance
        steps = self.steps
        option = self.option

        # Generate the filter
        if distance * steps * amplification != 0:  # The function is only called if there are no zeros (blurring).
            filter = self.__createFilter()

        # In this part, images are generated for each sample.
        for i in range(X.shape[0]):
            matrix_a = np.zeros(matrix.shape)
            if distance * steps * amplification != 0:  # The function is only called if there are no zeros (blurring).
                matrix_a = self.__blurringFilter(matrix_a, filter, X[i], coord)
            else:  # (no blurring)
                iter_values_X = iter(X[i])
                for eje_x, eje_y in coord:
                    matrix_a[int(eje_x), int(eje_y)] = next(iter_values_X)

            extension = 'png'  # eps o pdf
            subfolder = str(int(Y[i])).zfill(2)  # subfolder for grouping the results of each class
            name_image = str(i).zfill(6)
            route = os.path.join(folder, subfolder)
            route_complete = os.path.join(route, name_image + '.' + extension)
            if not os.path.isdir(route):
                try:
                    os.makedirs(route)
                except:
                    print("Error: Could not create subfolder")
            matplotlib.image.imsave(route_complete, matrix_a, cmap='binary', format=extension)

        return matrix

    def __obtainCoord(self, X):
        """
        This function uses the dimensionality reduction algorithm in order to represent the characteristic
        pixels in the image. The specifications of this function are:
        - Perform a normalisation of (0,1) to be able to represent the pixels inside the square.
        - Transpose the matrix.
        - Set the dimensionality reduction algorithm, PCA or t-SNE.
        """

        self.min_max_scaler = MinMaxScaler()
        X = self.min_max_scaler.fit_transform(X)

        labels = np.arange(X.shape[1])
        X_trans = X.T

        if (self.verbose):
            print("Selected algorithm: " + self.algorithm)

        if (self.algorithm == 'PCA'):
            X_embedded = PCA(n_components=2, random_state=self.seed).fit(X_trans).transform(X_trans)
        elif (self.algorithm == 't-SNE'):
            for i in range(self.times):
                X_trans = np.append(X_trans, X_trans, axis=0)
                labels = np.append(labels, labels, axis=0)
            X_embedded = TSNE(n_components=2, random_state=self.seed, perplexity=50).fit_transform(X_trans)
        else:
            print("Error: Incorrect algorithm")
            X_embedded = np.random.rand(X.shape[1], 2)

        data_coord = {'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'Label': labels}
        dc = pd.DataFrame(data=data_coord)
        self.obtain_coord = dc.groupby('Label').mean().values

        del X_trans
        gc.collect()

    def __areaDelimitation(self):
        """
        This function performs the delimitation of the area
        """
        self.initial_coordinates, self.vertices = self.__square(self.obtain_coord)

    def __matrixPositions(self, filename='original'):
        """
        This function gets the positions in the matrix
        """
        self.pos_pixel_caract, self.m, self.error_pos = self.__m_imagen(self.initial_coordinates, self.vertices)

    def __CrearImg(self, X, Y, folder='prueba/', train_m=False):
        """
        This function creates the images that will be processed by CNN.
        """

        X_scaled = self.min_max_scaler.transform(X)
        Y = np.array(Y)
        try:
            os.mkdir(folder)
            if self.verbose:
                print("The folder was created " + folder + "...")
        except:
            if self.verbose:
                print("The folder " + folder + " is already created...")

        self.m = self.__imageSampleFilter(X_scaled, Y, self.pos_pixel_caract, self.m, folder)

    def __trainingAlg(self, X, Y, folder='img_train/', verbose=False):
        """
        This function uses the above functions for the training.
        """
        self.__obtainCoord(X)
        self.__areaDelimitation()
        self.__matrixPositions()
        self.__CrearImg(X, Y, folder, train_m=True)

    def __testAlg(self, X, Y=None, folder='img_test/', verbose=False):
        """
        This function uses the above functions for the validation.
        """
        if (Y is None):
            Y = np.zeros(X.shape[0])
        self.__CrearImg(X, Y, folder, train_m=False, verbose=verbose)

    ###########################################################

    def generateImages(self):
        # Blurring verification
        if not self.blur:
            self.amplification = 0
            self.distance = 2
            self.steps = 0

        # Read the CSV
        dataset = pd.read_csv(self.src_data)
        array = dataset.values

        X = array[:, :-1]
        Y = array[:, -1]

        # Training
        self.__trainingAlg(X, Y, folder=self.dest_folder, verbose=self.verbose)
