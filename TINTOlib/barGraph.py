import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image

class BarGraph:
    default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
    default_verbose = False     # Verbose: if it's true, show the compilation text
    default_pixel_width = 1     # Width of the bars pixels
    default_gap = 0             # Gap between graph bars

    def __init__(
        self,
        verbose: bool = default_verbose,
        pixel_width: int = default_pixel_width,
        gap: int = default_gap,
        problem: str = default_problem
    ):
        """
        Arguments:
        verbose: bool
            Whether to display information about the progress
        pixel_width: int
            The width (in pixels) for each column
        gap: int
            The separation (in pixels) between each column.
        problem: str
            String representing the type of dataset
        """

        if not isinstance(pixel_width, int):
            raise TypeError(f"pixel_width must be of type int (got {type(pixel_width)})")
        if pixel_width <= 0:
            raise ValueError(f"pixel_width must be positive (got {pixel_width})")
        if not isinstance(gap, int):
            raise TypeError(f"gap must be of type int (got {type(gap)})")
        if pixel_width < 0:
            raise ValueError(f"gap cannot be negative (got {gap})")
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

        route_relative = os.path.join(subfolder, name_image)
        return route_relative

    def __trainingAlg(self, X, Y):
        imagesRoutesArr = []    # List to store the routes

        # Image variables
        n_columns = X.shape[1]
        
        # There is a gap before the first column and after all the columns (n_columns columns & n_colums+1 gaps)
        image_size = self.pixel_width*n_columns + (n_columns+1)*self.gap
        # Add a padding on top and bottom
        top_padding, bottom_padding = self.pixel_width, self.pixel_width
        max_bar_height = image_size - (bottom_padding + top_padding)
        # Step of column (width + gap)
        step_column = self.gap + self.pixel_width

        for i,sample in enumerate(X):
            # Create the image (the image is squared)
            image = np.zeros([image_size, image_size, 1])
            # Multiply the values in the sample time the height of the bar
            bar_heights = np.floor(sample * max_bar_height).astype(np.int64)
            for i_bar,val_bar in enumerate(bar_heights):
                image[
                    top_padding : (top_padding + val_bar),                                         # The height of the column
                    (self.gap+(step_column*i_bar)) : (self.gap+(step_column*i_bar)) + self.pixel_width # The width of the column
                ] = 1
                
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

        Arguments
        ---------
        data: data CSV or pandas Dataframe
            The data and targets
        folder: str
            The folder where the images are created
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

        # Normalize the data
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        X = (X - min_vals) / (max_vals - min_vals)

        # TODO: reorder columns

        # Training
        self.__trainingAlg(X, Y)
