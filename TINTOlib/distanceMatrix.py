from TINTOlib.abstractImageMethod import AbstractImageMethod
import os
import pandas as pd
import numpy as np
from PIL import Image
from typing import Union

class DistanceMatrix(AbstractImageMethod):
    default_zoom = 1                # Scale of the image 1:x

    def __init__(
        self,
        problem = None,
        verbose = None,
        zoom=default_zoom,
    ):
        super().__init__(problem=problem, verbose=verbose)
        self.zoom = zoom

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

    def _trainingAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        X = x.values
        Y = y.values if y is not None else None

        imagesRoutesArr = []
        N,d=X.shape

        #Create matrix (only once, then reuse it)
        imgI = np.empty((d,d))

        #For each instance
        for ins,dataInstance in enumerate(X):
            for i in range(d):
                for j in range(d):
                    imgI[i][j] = dataInstance[i]-dataInstance[j]

            #Normalize matrix
            image_norm = (imgI - np.min(imgI)) / (np.max(imgI) - np.min(imgI))
            image = np.repeat(np.repeat(image_norm, self.zoom, axis=0), self.zoom, axis=1)

            if self.problem == "supervised":
                route = self.__saveSupervised(Y[ins], ins, image)
                imagesRoutesArr.append(route)
            elif self.problem == "unsupervised" or self.problem == "regression":
                route = self.__saveRegressionOrUnsupervised(ins, image)
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
