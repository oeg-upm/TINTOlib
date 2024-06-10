import pickle
import os
import matplotlib
import pandas as pd
import numpy as np
class BaseModel:
    default_verbose = False  # Verbose: if it's true, show the compilation text
    def __init__(self, verbose=default_verbose):
        self.verbose = verbose

    def saveHyperparameters(self, filename='objs'):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print("It has been successfully saved in " + filename)

    def loadHyperparameters(self, filename='objs.pkl'):
        """
        This function allows LOADING the transformation options to images from a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
        
        for key, val in variables.items():
            setattr(self, key, val)

        if self.verbose:
            print("It has been successfully loaded from " + filename)


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


    def __createImage(self, X, Y, folder='prueba/', train_m=False):
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

    def __trainingAlg(self, X, Y, folder='img_train/'):
        """
        This function uses the above functions for the training.
        """
        self.__obtainCoord(X)
        self.__areaDelimitation()
        self.__matrixPositions()
        self.__createImage(X, Y, folder, train_m=True)

    def generateImages(self,data, folder):
        """
            This function generate and save the synthetic images in folders.
                - data : data CSV or pandas Dataframe
                - folder : the folder where the images are created
        """
        # Blurring verification

        # Read the CSV
        if type(data) == str:
            dataset = pd.read_csv(data)
            array = dataset.values
        elif isinstance(data,pd.DataFrame) :
            array = data.values
        X = array[:, :-1]
        Y = array[:, -1]

        # Training
        self.__trainingAlg(X, Y, folder=folder)
