from __future__ import division
import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class SuperTML:
    ###### default values ###############
    default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
    default_verbose = False  # Verbose: if it's true, show the compilation text
    def __init__(self, problem=default_problem, verbose=default_verbose):
        self.problem = problem
        self.verbose = verbose
    def __event2img(self,event: np.ndarray, size: int = 224) -> Image.Image:
        #font = ImageFont.truetype('Arial.ttf', 13)
        font = ImageFont.load_default()
        img = Image.fromarray(np.zeros([size, size, 3]), 'RGB')
        for i, f in enumerate(event):
            ImageDraw.Draw(img).text(((0.25 + (i % 2)) * size // 2, (i // 2) * 2 * size // len(event)), f'{f:.3f}',
                                     fill=(255, 255, 255), font=font)
        return img

    def __saveSupervised(self, Y, i, folder, image):
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
        image.save(route_complete)

    def __saveUnsupervised(self, i, folder, image):
        extension = 'png'  # eps o pdf
        name_image = str(i).zfill(6)
        route = os.path.join(folder)
        route_complete = os.path.join(route, name_image + '.' + extension)
        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")
        image.save(route_complete)

    def __saveRegression(self, value, i, folder, image):
        extension = 'png'  # eps o pdf
        subfolder = "images"
        name_image = str(i).zfill(6) + '(' + str(value) + ')' + '.' + extension
        route = os.path.join(folder, subfolder)
        route_complete = os.path.join(route, name_image)
        # route_complete = os.path.join(route, name_image +'.' + extension)
        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")
        image.save(route_complete)

        route_relative = os.path.join(subfolder, name_image)
        return route_relative


    def __createImages(self, X, Y):
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
            """extension = 'png'  # eps o pdf
            subfolder = str(int(Y[i])).zfill(2)  # subfolder for grouping the results of each class
            name_image = str(i).zfill(6)
            route = os.path.join(self.folder, subfolder)
            route_complete = os.path.join(route, name_image + '.' + extension)
            if not os.path.isdir(route):
                try:
                    os.makedirs(route)
                except:
                    print("Error: Could not create subfolder")"""
            image=self.__event2img(X[i])
            #image.save(route_complete)

            if self.problem == "supervised":
                self.__saveSupervised(Y,i,self.folder,image)
            elif self.problem == "unsupervised":
                self.__saveUnsupervised(i,self.folder,image)
            elif self.problem == "regression":
                value = Y[i]
                route = self.__saveRegression(value, i, self.folder, image)
                imagesRoutesArr.append(route)
            else:
                print("Wrong problem definition. Please use 'supervised', 'unsupervised' or 'regression'")



    def __trainingAlg(self, X, Y):
        #font = ImageFont.truetype('./arial.ttf', 13)
        self.__createImages(X,Y)
    def generateImages(self,data, folder="prueba/"):
        """
            This function generate and save the synthetic images in folders.
                - data : data CSV or pandas Dataframe
                - folder : the folder where the images are created
        """
        # Read the CSV
        self.folder=folder
        if type(data)==str:
            dataset = pd.read_csv(data)
            array = dataset.values
        elif type(data)==pd:
            array = data.values

        X = array[:, :-1]
        Y = array[:, -1]

        # Training
        self.__trainingAlg(X, Y)
