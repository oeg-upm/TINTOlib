# Standard library imports
import os

# Third-party library imports
import numpy as np
import pandas as pd
from PIL import Image

# Typing imports
from typing import Optional, Union

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    BarGraph    ###########################
###########################################################

class BarGraph(AbstractImageMethod):
    """
    BarGraph: Represents normalized variable values within [0, 1] in a bar graph format.

    This method visualizes data as a bar graph, where each variable is represented as a bar with a configurable 
    width and gap between bars.

    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        Default is 'supervised'. Valid values: ['supervised', 'unsupervised', 'regression'].
    normalize : bool, optional
        If True, normalizes input data using MinMaxScaler. 
        Default is True. Valid values: [True, False].
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    pixel_width : int, optional
        The width of each bar in pixels. 
        Default is 1. Valid values: integer > 0.
    gap : int, optional
        The gap between bars in pixels. 
        Default is 0. Valid values: integer >= 0.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    """
    ###### Default values ###############
    default_pixel_width = 1  # Width of the bars in pixels
    default_gap = 0  # Gap between the bars
    default_zoom = 1  # Rescale factor for saving the image

    def __init__(
        self,
        problem = None,
        normalize=None,
        verbose = None,
        pixel_width: int = default_pixel_width,
        gap: int = default_gap,
        zoom: int = default_zoom,
    ):
        super().__init__(problem=problem, verbose=verbose, normalize=normalize)

        if not isinstance(pixel_width, int):
            raise TypeError(f"pixel_width must be of type int (got {type(pixel_width)})")
        if pixel_width <= 0:
            raise ValueError(f"pixel_width must be positive (got {pixel_width})")
        if not isinstance(gap, int):
            raise TypeError(f"gap must be of type int (got {type(gap)})")
        if pixel_width < 0:
            raise ValueError(f"gap cannot be negative (got {gap})")
        
        self.pixel_width = pixel_width
        self.gap = gap

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
        img = img.resize(size=(img.size[0]*self.zoom, img.size[1]*self.zoom), resample=Image.Resampling.NEAREST)
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
        img = img.resize(size=(img.size[0]*self.zoom, img.size[1]*self.zoom), resample=Image.Resampling.NEAREST)
        img.save(route_complete)

        route_relative = os.path.join(subfolder, name_image)
        return route_relative
    
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Fit method for stateless transformers. Does nothing and returns self.
        """
        return self
    
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        x = x.values
        Y = y.values if y is not None else None

        # TODO: reorder columns

        imagesRoutesArr = []    # List to store the routes

        # Image variables
        n_columns = x.shape[1]
        
        # There is a gap before the first column and after all the columns (n_columns columns & n_colums+1 gaps)
        image_size = self.pixel_width*n_columns + (n_columns+1)*self.gap
        # Add a padding on top and bottom
        top_padding, bottom_padding = self.pixel_width, self.pixel_width
        max_bar_height = image_size - (bottom_padding + top_padding)
        # Step of column (width + gap)
        step_column = self.gap + self.pixel_width

        for i,sample in enumerate(x):
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
