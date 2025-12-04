# Standard library imports
import gc
import math
import os

# Third-party library imports
import matplotlib
import matplotlib.image
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Typing imports
from typing import Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    TINTO    ##############################
###########################################################


class TINTO(AbstractImageMethod):
    """
    TINTO: A class for transforming tabular data into synthetic images by projecting it into a two-dimensional space 
    and applying visualization techniques.
    
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
    pixels : int, optional
        The number of pixels used to create the image (one side). 
        Total pixels = pixels * pixels. Default is 20. Valid values: integer.
    algorithm : str, optional
        Select the dimensionality reduction algorithm. 
        Default is 'PCA'. Valid values: ['PCA', 't-SNE'].
    blur : bool, optional
        Activate or deactivate the blurring option. 
        Default is False. Valid values: [True, False].
    submatrix : bool, optional
        Specifies whether to use a submatrix for blurring. 
        Default is True. Valid values: [True, False].
    amplification : float, optional
        Only used when `blur=True`. Specifies the blurring amplification. 
        Default is `np.pi`. Valid values: float.
    distance : int, optional
        Only used when `blur=True`. Specifies the blurring distance (number of pixels). 
        Default is 2. Valid values: integer.
    steps : int, optional
        Only used when `blur=True`. Specifies the number of blurring steps. 
        Default is 4. Valid values: integer.
    option : str, optional
        Only used when `blur=True`. Technique for handling overlapping pixels. 
        Default is 'mean'. Valid values: ['mean', 'maximum'].
    times : int, optional
        Only used when `algorithm='t-SNE'`. Specifies the replication times in t-SNE. 
        Default is 4. Valid values: integer.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Values greater than 1 increase the image size proportionally. 
        Default is 1. Valid values: integer.
    random_seed : int, optional
        Seed for reproducibility. 
        Default is 1. Valid values: integer.

    Attributes:
    ----------
    error_pos : bool
        Indicates overlap of characteristic pixels during image creation.
    """
    ###### default values ###############
    default_pixels = 20  # Image's Pixels (one side)

    default_algorithm = "PCA"  # Dimensionality reduction algorithm (PCA or t-SNE)
    default_blur = False  # Activate blurring option

    default_submatrix = True  # Use or not use submatrix
    default_amplification = np.pi  # Amplification factor in blurring
    default_distance = 2  # Distance in blurring (number of pixels)
    default_steps = 4  # Steps in blurring
    default_option = 'mean'  # Option in blurring ('mean' or 'maximum')

    default_times = 4  # Replication times in t-SNE
    default_train_m = True  # Use training matrix

    default_random_seed = 1  # Seed for reproducibility
    default_zoom = 1  # Zoom level

    def __init__(
        self,
        problem=None,
        transformer=MinMaxScaler(),
        verbose=None,
        pixels=default_pixels,
        algorithm=default_algorithm,
        blur=default_blur,
        submatrix=default_submatrix,
        amplification=default_amplification,
        distance=default_distance,
        steps=default_steps,
        option=default_option,
        times=default_times,
        train_m=default_train_m,
        zoom=default_zoom,
        random_seed=default_random_seed,
    ):
        super().__init__(problem=problem, verbose=verbose, transformer=transformer)

        self.pixels = pixels

        self.algorithm = algorithm
        self.blur = blur
        
        self.submatrix = submatrix
        self.amplification = amplification
        self.distance = distance
        self.steps = steps
        self.option = option

        self.times = times
        self.train_m = train_m

        self.zoom = zoom
        self.random_seed = random_seed

        self.error_pos = False  # Indicates the overlap of characteristic pixels.

    def __square(self, coord):
        """
        This function has the purpose of being able to create the square delimitation of the resulting image.
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

        if np.count_nonzero(matrix != 0) != coord.shape[0]:
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
                    if (center_x - i) ** 2 + (center_y - j) ** 2 <= r_actual ** 2:
                        filter[i, j] = intensity
        filter[center_x, center_y] = 1
        return filter

    def __blurringFilterSubmatrix(self, matrix, filter, values, coordinates):
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

    def __blurringFilter(self, matrix, coordinates):
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

        x = int(coordinates[1])
        y = int(coordinates[0])
        valor_central = matrix[x, y]

        for p in range(self.steps):
            r_actual = int(matrix.shape[0] * self.distance * (p + 1))  # radio actual  de mayor a menor

            # Funcion de intensidad
            intensidad = min(self.amplification * valor_central / (np.pi * r_actual ** 2), valor_central)

            # Delimitación del área
            lim_inf_i = max(x - r_actual - 1, 0)
            lim_sup_i = min(x + r_actual + 1, matrix.shape[0])
            lim_inf_j = max(y - r_actual - 1, 0)
            lim_sup_j = min(y + r_actual + 1, matrix.shape[1])

            # Colocación de valores
            for i in range(lim_inf_i, lim_sup_i):
                for j in range(lim_inf_j, lim_sup_j):
                    if ((x - i) ** 2 + (y - j) ** 2 <= r_actual ** 2):
                        if (matrix[i, j] == 0):
                            matrix[i, j] = intensidad
                        elif (x != i and y != j):  # Sobreposición
                            if (self.option == 'mean'):
                                matrix[i, j] = (matrix[i, j] + intensidad) / 2
                            elif (self.option == 'maximum'):
                                matrix[i, j] = max(matrix[i, j], intensidad)
        return matrix




    def __imageSampleFilterSubmatrix(self, X, Y, coord, matrix):
        """
        This function creates the samples, i.e., the images. This function has the following specifications:
        - The first conditional performs the pre-processing of the images by creating the matrices.
        - Then the for loop generates the images for each sample.
        """
        # Hyperparams
        amplification = self.amplification
        distance = self.distance
        steps = self.steps
        option = self.option
        # Generate the filter
        if distance * steps * amplification != 0:  # The function is only called if there are no zeros (blurring).
            filter = self.__createFilter()

        total = X.shape[0]

        # In this part, images are generated for each sample.
        for i in range(total):
            matrix_a = np.zeros(matrix.shape)
            if distance * steps * amplification != 0:  # The function is only called if there are no zeros (blurring).
                matrix_a = self.__blurringFilterSubmatrix(matrix_a, filter, X[i], coord)
            else:  # (no blurring)
                iter_values_X = iter(X[i])
                for eje_x, eje_y in coord:
                    matrix_a[int(eje_x), int(eje_y)] = next(iter_values_X)

            matrix_a = np.repeat(np.repeat(matrix_a, self.zoom, axis=0), self.zoom, axis=1)
            self._save_image(matrix_a,Y[i], i)

        return matrix

    def __imageSampleFilter(self, X, Y, coord, matrix):
        """
        This function creates the samples, i.e., the images. This function has the following specifications:
        - The first conditional performs the pre-processing of the images by creating the matrices.
        - Then the for loop generates the images for each sample.
        """
        total = X.shape[0]
        # Aquí es donde se realiza el preprocesamiento siendo 'matriz' = 'm'
        if self.train_m:

            matrix_a = np.zeros(matrix.shape)

            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    matrix_a[int(coord[j, 1]), int(coord[j, 0])] = X[i, j]
                    matrix_a = self.__blurringFilter(matrix_a, coord[j] )

            matriz = np.copy(matrix_a)
        if self.verbose:
            self.bar.write("Generating images......")
        # En esta parte se generan las imágenes por cada muestra
        for i in range(total):
            matrix_a = np.copy(matriz)

            for j in range(X.shape[1]):
                matrix_a[int(coord[j, 1]), int(coord[j, 0])] = X[i, j]
                matrix_a = self.__blurringFilter(matrix_a, coord[j])

            for j in range(X.shape[1]):
                matrix_a[int(coord[j, 1]), int(coord[j, 0])] = X[i, j]

            matrix_a = np.repeat(np.repeat(matrix_a, self.zoom, axis=0), self.zoom, axis=1)
            self._save_image(matrix_a, Y[i], i)

            # Verbose
            if self.verbose:
                self.bar.write("Created ", str(i + 1), "/", int(total))


        return matrix

    def __obtainCoord(self, X):
        """
        This function uses the dimensionality reduction algorithm in order to represent the characteristic
        pixels in the image. The specifications of this function are:
        - Perform a normalisation of (0,1) to be able to represent the pixels inside the square.
        - Transpose the matrix.
        - Set the dimensionality reduction algorithm, PCA or t-SNE.
        """

        labels = np.arange(X.shape[1])
        
        X_trans = X.T

        if self.verbose:
            self.bar.write("Selected algorithm: " + self.algorithm)

        if self.algorithm == 'PCA':
            X_embedded = PCA(n_components=2, random_state=self.random_seed).fit(X_trans).transform(X_trans)
        elif self.algorithm == 't-SNE':
            for i in range(self.times):
                X_trans = np.append(X_trans, X_trans, axis=0)
                labels = np.append(labels, labels, axis=0)
            X_embedded = TSNE(n_components=2, random_state=self.random_seed, perplexity=50).fit_transform(X_trans)
        else:
            self.bar.write("Error: Incorrect algorithm")
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

    def __matrixPositions(self):
        """
        This function gets the positions in the matrix
        """
        self.pos_pixel_caract, self.m, self.error_pos = self.__m_imagen(self.initial_coordinates, self.vertices)

    def __createImage(self, X, Y, folder='prueba/'):
        """
        This function creates the images.
        """
        X = X.values
        Y = np.array(Y)
        try:
            os.makedirs(folder)
            if self.verbose:
                self.bar.write("The folder was created " + folder + "...")
        except:
            if self.verbose:
                self.bar.write("The folder " + folder + " is already created...")

        if self.submatrix:
            self.m = self.__imageSampleFilterSubmatrix(X, Y, self.pos_pixel_caract, self.m)
        else:
            self.m = self.__imageSampleFilter(X, Y, self.pos_pixel_caract, self.m)

    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        if not self.blur:
            self.amplification = 0
            self.distance = 2
            self.steps = 0

        X = x.values
        Y = y.values if y is not None else None

        self.__obtainCoord(X)
        self.__areaDelimitation()
        self.__matrixPositions()

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        if not self.blur:
            self.amplification = 0
            self.distance = 2
            self.steps = 0

        if (y is None):
            y = np.zeros(x.shape[0])
        self.__createImage(x, y, self.folder)
        self._features_pos_to_csv(x.columns,self.pos_pixel_caract)

    def _img_to_file(self, image_matrix, file,extension):
        matplotlib.image.imsave(file, image_matrix, cmap='binary', format=extension, dpi=self.zoom)
