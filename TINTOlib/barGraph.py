import pickle
import os
import matplotlib
import pandas as pd
import numpy as np
from PIL import Image
class BarGraph:
    default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
    default_verbose = False     # Verbose: if it's true, show the compilation text
    default_pixel_width = 3     # Width of the bars pixels
    default_gap = 2            # Gap between graph bars
    def __init__(self, verbose=default_verbose, pixel_width=default_pixel_width,gap=default_gap,problem=default_problem):
        self.problem=problem
        self.verbose = verbose
        self.pixel_width = pixel_width
        self.gap = gap
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
        This function allows LOADING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)

        if self.verbose:
            print("It has been successfully loaded in " + filename)
    def __saveSupervised(self, y, i, image):
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

        img = Image.fromarray(np.uint8(np.squeeze(image) * 255))
        img.save(route_complete)

        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __saveRegressionOrUnsupervised(self, i, image):
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
        img = Image.fromarray(np.uint8(np.squeeze(image) * 255))
        img.save(route_complete)

        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __trainingAlg(self, X, Y):
        """
        This function uses the above functions for the training.
        """

        imagesRoutesArr = []
        # Image variables
        px=self.pixel_width
        gap=self.gap
        N,d=X.shape
        img_sz = [(px * d + gap * (d)), (px * d + gap * (d))]
        max_bar_height = img_sz[0]   # leave some space (padding) for bottom and up.

        for i in range(N):
            barI = np.floor(max_bar_height  * X[i, :]).astype(int)
            k = 0
            image = np.zeros([img_sz[0], img_sz[1], 1])

            # upside down images will be created
            for j in range(0 , img_sz[1] - gap , gap+px):
                image[px:barI[k], j:j + px, :] = 1
                k = k + 1
                if k > d:
                    break

            if self.problem == "supervised":
                route = self.__saveSupervised(Y[i], i, image)
                imagesRoutesArr.append(route)
            elif self.problem == "unsupervised" or self.problem == "regression":
                route = self.__saveRegressionOrUnsupervised(i, image)
                imagesRoutesArr.append(route)
            else:
                print("Wrong problem definition. Please use 'supervised', 'unsupervised' or 'regression'")

        if self.problem == "supervised":
            data = {'images': imagesRoutesArr, 'class': Y}
            supervisedCSV = pd.DataFrame(data=data)
            supervisedCSV.to_csv(self.folder + "/supervised.csv", index=False)
        elif self.problem == "unsupervised":
            data = {'images': imagesRoutesArr}
            unsupervisedCSV = pd.DataFrame(data=data)
            unsupervisedCSV.to_csv(self.folder + "/unsupervised.csv", index=False)
        elif self.problem == "regression":
            data = {'images': imagesRoutesArr, 'values': Y}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/regression.csv", index=False)

    def generateImages(self,data, folder="img_train/"):
        """
            This function generate and save the synthetic images in folders.
                - data : data CSV or pandas Dataframe
                - folder : the folder where the images are created
        """
        # Read the CSV
        self.folder = folder
        if type(data) == str:
            dataset = pd.read_csv(data)
            array = dataset.values
        elif isinstance(data,pd.DataFrame) :
            array = data.values
        X = array[:, :-1]
        Y = array[:, -1]

        # Training
        self.__trainingAlg(X, Y)
