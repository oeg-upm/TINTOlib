from __future__ import division
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import math 

class SuperTML:
    ###### default values ###############
    default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
    default_verbose = False  # Verbose: if it's true, show the compilation text
    default_columns = 4 #Number of columns
    default_size = 224
    default_font_size = 10
    def __init__(self, problem=default_problem, verbose=default_verbose, columns=default_columns,
                 size=default_size, font_size = default_font_size):

        self.problem = problem
        self.verbose = verbose
        self.columns = columns
        self.image_size = size
        self.font_size = font_size

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

            self.problem = variables["problem"]
            self.verbose = variables["verbose"]
            self.columns = variables["columns"]
            self.image_size = variables["image_size"]
            self.font_size = variables["font_size"]

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

        image.save(route_complete)
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
        image.save(route_complete)

        route_relative = os.path.join(subfolder, name_image)
        return route_relative

    def __event2img(self,event: np.ndarray):
        cell_width = self.image_size // self.columns
        rows = math.ceil(len(event) / self.columns)
        cell_height = self.image_size // rows

        font = ImageFont.truetype("arial.ttf", self.font_size)
        img = Image.fromarray(np.zeros([self.image_size, self.image_size, 3]), 'RGB')
        draw = ImageDraw.Draw(img)

        for i, f in enumerate(event):
            x = ((i % self.columns)) * cell_width
            y = (i // self.columns) * cell_height

            text = f'{f:.3f}'
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x + (cell_width - text_width) / 2
            text_y = y + (cell_height - text_height) / 2

            draw.text(
                (text_x, text_y),
                text,
                fill=(255, 255, 255),
                font=font,
            )
        return img
    
    def __trainingAlg(self, X, Y):
        """
                This function creates the images that will be processed by CNN.
        """
        # Variable for regression problem
        imagesRoutesArr = []

        Y = np.array(Y)
        try:
            os.mkdir(self.folder)
            if self.verbose:
                print("The folder was created " + self.folder + "...")
        except:
            if self.verbose:
                print("The folder " + self.folder + " is already created...")
        for i in range(X.shape[0]):

            image = self.__event2img(X[i])

            if self.problem == "supervised":
                route = self.__saveSupervised(Y[i], i, image)
                imagesRoutesArr.append(route)
            elif self.problem == "unsupervised" or self.problem == "regression":
                route = self.__saveRegressionOrUnsupervised(i, image)
                imagesRoutesArr.append(route)
            else:
                print("Wrong problem definition. Please use 'supervised', 'unsupervised' or 'regression'")


    def generateImages(self,data, folder="prueba/"):
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
            elif isinstance(data, pd.DataFrame):
                array = data.values

            X = array[:, :-1]
            Y = array[:, -1]
            
            # Training
            self.__trainingAlg(X, Y)
            if self.verbose:
                print("End")
