import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TINTOlib.abstractImageMethod import AbstractImageMethod

###########################################################
################    MappingMethod    ##############################
###########################################################

default_format='png'
default_zoom=1

class MappingMethod(AbstractImageMethod):
    """
       MappingMethod: Abstract class that group similar functionality about mapping methods. Mapping methods are those that assign a pixel to each feature like
       TINTO,IGTD,REFINED,FOTOMIC and DEEPINSIGHT. This class inherits from AbstractImageMethod.

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
        zoom : int, optional
            Multiplication factor determining the size of the saved image relative to the original size.
            Default is 1. Valid values: integer > 0.
       format : str, optional
           Output format using images with matplotlib with [0,255] range for pixel or using npy format.
           Default is images with format 'png'.
       cmap : str, optional
           color map to use with matplotlib.

       Attributes:
       ----------
       error_pos : bool
           Indicates overlap of characteristic pixels during image creation.
       """
    def __init__(self, problem, verbose=False, transformer=None,format=default_format,zoom=default_zoom,cmap=None):
        super().__init__(problem=problem, verbose=verbose, transformer=transformer,format=format)
        self._features_mapping = None
        self.cmap=cmap
        self.zoom=zoom
        if(self.cmap!=None and self.cmap not in plt.colormaps()):
            raise Exception('cmap must be in matplotlib colormaps')

    def _build_features_mapping(self, features, features_positions):
        """
        This method builds a dataframe with the name of each column and its position on the image
        Args:
            features: Features Names
            features_positions: Pixel for each feature

        Returns:

        """
        self._features_mapping = pd.DataFrame(
            {"feature": features, "row": features_positions[:, 0], "column": features_positions[:, 1]})

    def _features_mapping_to_csv(self):
        """
        This method creates a csv file with the names of each column and its position on the image
        Returns:

        """
        filepath = self.folder + "/features_positions.csv"
        self._features_mapping.to_csv(filepath, index=False)

    def _get_features_mapping(self, features=None, columns_names=True):
        """
        This method return the positions in the image for the features specified.
        Args:
            features: List of features to get its positions
            columns_names: Indicate if input is a list of feature names or features positions on dataset

        Returns:
            dataframe of features positions
        """
        if (self._fitted == True and not self._features_mapping is None):
            if(features == None):
                return self._features_mapping
            else:
                if(columns_names):
                    return self._features_mapping[self._features_mapping.iloc[:,0].isin(features)]
                else:
                    return self._features_mapping.iloc[features]

    def _img_to_file(self,image_matrix,file):
        if(self.zoom!=1):
            image_matrix = np.repeat(np.repeat(image_matrix, self.zoom, axis=0), self.zoom, axis=1)


        if(self.format=='npy'):
            np.save(file, image_matrix.astype(np.float64))
        else:
            plt.imsave(file, image_matrix, cmap=self.cmap, format=self.format)