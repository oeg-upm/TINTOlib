from abc import abstractmethod
from typing import Union
import numpy as np
import pandas as pd
from TINTOlib.utils.assigner import AssignerFactory
import TINTOlib.utils.constants as constants
from TINTOlib.mappingMethod import MappingMethod
from sklearn.preprocessing import MinMaxScaler

###########################################################
################    ParamImageMethod    ##############################
###########################################################

default_random_seed = 1,
default_zoom=1
default_cmap = 'gray'  # Default cmap image output
group_methods_allowed=[constants.avg_option,constants.relevance_option]
opt_algorithms_allowed=[constants.linear_sum_assigner,constants.greedy_assigner]
assigners_allowed=[constants.bin_assigner,constants.bin_digitize_assigner,constants.quantile_assigner,constants.pixel_centroids_assigner,constants.relevance_assigner]

class ParamImageMethod(MappingMethod):
    """
       MappingMethod: Abstract class that group similar functionality about mapping parametric methods. Mapping parametric methods are those that assign a pixel to each feature using
        an extraction features coordinates process following by a mapping process like FOTOMIC and DEEPINSIGHT. This class inherits from MappingMethod and AbstractImageMethod.

       Parameters:
       ----------
       dim : int
           Order size for a square matrix image
       problem : str, optional
           The type of problem, defining how the images are grouped.
           Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
      transformer : CustomTransformer, optional
           Preprocessing transformations like scaling, normalization,etc.
           Default is LogScaler custom transformer.
           Valid: Scikit Learn transformers or custom implementation using inheritance over CustomTransformer class.
       verbose : bool, optional
           Show execution details in the terminal.
           Default is False. Valid values: [True, False].
       assignment_method : str, optional
           Using to apply different techniques to mapping features with pixels. Default is 'bin'.
       relocate: bool, optional
           Relocate features so that each pixel can represent a single feature. Use the relevance of features and pixels shift to build a cost function
       algorithm_opt : str, optional
           Optimization algorithm that could be apply in pixels assignment stage.
       group_method : str, optional
           Using to apply different techniques to calculate pixels values that share multiples features. Default is 'avg'.
       zoom : int, optional
            Multiplication factor determining the size of the saved image relative to the original size.
            Default is 1. Valid values: integer > 0.
       format : str, optional
           Output format using images with matplotlib with [0,255] range for pixel or using npy format.
           Default is images with format 'png'.
       cmap : str, optional
           color map to use with matplotlib.
           Default is gray
       random_seed : int, optional
           Seed for reproducibility.
           Default is 1. Valid values: integer.
       """
    def __init__(self,
            dim,
            problem=None,
            transformer=None,
            verbose=None,
            assignment_method=constants.bin_assigner,
            relocate=False,
            algorithm_opt=constants.linear_sum_assigner,
            group_method=constants.avg_option,
            zoom=default_zoom,
            format=constants.png_format,
            cmap=default_cmap,
            random_seed=default_random_seed
    ):
        if (assignment_method not in assigners_allowed):
            raise ValueError(f"Algorithm_rd parameter must be in {assigners_allowed}")

        if (algorithm_opt not in opt_algorithms_allowed):
            raise ValueError(f"Algorithm_rd parameter must be in {opt_algorithms_allowed}")

        if (group_method not in group_methods_allowed):
            raise ValueError(f"Group method must be in {group_methods_allowed}")

        super().__init__(problem=problem, verbose=verbose, transformer=transformer,format=format, zoom=zoom, cmap=cmap)
        self._image_dim = dim
        self._algorithm_opt = algorithm_opt
        self._random_seed = random_seed
        self._assignment_method = assignment_method
        self._relocate = relocate
        self._group_method=group_method


    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
               Mapping each feature with a pixel in a image with a specific size. The final matrix is assigned to class variable to use later in transform process.
               Args:
                   x: independent variables set (features)
                   y: class variable

               Returns:

               """
        x_transposed = x.T
        self._features_coord=self._get_features_coords(x_transposed)
        if(self._relocate or self._group_method==constants.relevance_option):
            self._features_relevance = self._compute_relevance(x,self._features_coord)

        # Mapping features coordinates with image
        self._mapping_features(x)
        self._build_features_mapping(x.columns,self._features_positions)

    def _mapping_features(self,x):
        assigner = AssignerFactory.get_assigner(self._assignment_method,
                                                algorithm=self._algorithm_opt)
        self._features_positions = assigner.assign(self._features_coord, self._image_dim)
        if (self._relocate):
            if (self._duplicated(self._features_positions)):
                optimizer = AssignerFactory.get_assigner(constants.relevance_assigner, algorithm=self._algorithm_opt)
                self._features_positions = optimizer.assign(np.hstack((self._features_positions,self._features_relevance)), self._image_dim)


    def _duplicated(self, features_positions):
        """

        Args:
            features_positions: array with pixels positions by feature

        Returns:
            Boolean indicating if features positions are duplicated
        """
        df_feature_pos = pd.DataFrame(features_positions)
        return df_feature_pos.duplicated().any()



    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """

        Args:
            x: Pandas Dataframe with feature tabular data
            y: Pandas Dataframe with class tabular data

        """
        imgs_coord = self._calculate_pixels_values(x)
        # Create and Save Images
        self._create_images(imgs_coord, y)
        self._features_mapping_to_csv()


    def _calculate_pixels_values(self,x):
        """
        Calls the specified method for calculating the value of each pixel.
        Args:
            x: Dataset

        Returns:
            imgs_coord: Matrix with pixel values by averaging features values
        """
        match self._group_method:
            case constants.avg_option:
                return self._avg_features_by_pixels(x)
            case constants.relevance_option:
                return self._rev_features_by_pixels(x)
            case _:
                return self._avg_features_by_pixels(x)


    def _avg_features_by_pixels(self,x):
        """
        Calculate the pixel values by averaging features values that correspond to each pixel
        Args:
            x: Dataset

        Returns:
            imgs_coord: Matrix with pixel values by averaging features values
        """
        imgs_coord = (pd.DataFrame(np.vstack((self._features_positions.T,x)).T)
                      .groupby([0, 1], as_index=False).mean())
        return imgs_coord

    def _rev_features_by_pixels(self,x):
        """
        Calculate the pixel values by weighted sum of the value of each feature and its relevance
        Args:
            x:Dataset

        Returns:

        """

        features_info=np.hstack((self._features_positions, self._features_relevance))
        features_dot_rev=(x.T)*(features_info[:,2].reshape(-1,1))
        rev_sum = (pd.DataFrame(features_info).groupby([0, 1], as_index=False).transform('sum')).to_numpy()
        features_rev=features_dot_rev/rev_sum
        pixels_sum=(pd.DataFrame(np.hstack((self._features_positions, features_rev))).groupby([0, 1], as_index=False).sum()).to_numpy()
        rev_count = (pd.DataFrame(features_info).groupby([0, 1], as_index=False).count()).to_numpy()
        pixels_values=pixels_sum[:,2:]/rev_count[:,2].reshape(-1,1)
        imgs_coord=pd.DataFrame(np.hstack((pixels_sum[:,:2], pixels_values)))
        return imgs_coord


    def _create_images(self,imgs_coord,y):
        """
        Create images matrix by fill the pixel coordinates without feature assigned to 0. In other case, fill the calculate value. Save the image file and create
        csv file.
        Args:
            imgs_coord: Matrix with pixel values by averaging features values
            y: class variable

        Returns:

        """
        for m in range(2,imgs_coord.shape[1]):
            i=m-2
            img=np.zeros((self._image_dim,self._image_dim))
            img[imgs_coord[0].astype(int),imgs_coord[1].astype(int)]=imgs_coord[m]
            self._save_image(img,y.iloc[i],i)

    @abstractmethod
    def _get_features_coords(self,x):
        raise NotImplementedError("Subclasses must implement _fit_alg.")

    @abstractmethod
    def _compute_relevance(self, x=None,features_coord=None):
        raise NotImplementedError("Subclasses must implement _compute_relevance.")
