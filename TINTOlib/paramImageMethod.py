from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from TINTOlib.abstractImageMethod import AbstractImageMethod
from TINTOlib.utils.assigner import AssignerFactory
import matplotlib.image

default_assignment_method = 'bin'
default_random_seed = 1,
default_algorithm_opt = 'lsa'
default_group_method = 'avg'

class ParamImageMethod(AbstractImageMethod):

    def __init__(self,
            dim,
            problem=None,
            transformer=None,
            verbose=None,
            assignment_method=default_assignment_method,
            relocate=False,
            algorithm_opt=default_algorithm_opt,
            group_method=default_group_method,
            random_seed=default_random_seed
    ):
        if (assignment_method not in ["bin", "quantile_transform", "PixelCentroidsAssigner","binDigitize"]):
            raise ValueError("Algorithm_rd parameter must be in [bin,quantile_transform,PixelCentroidsAssigner,binDigitize]")

        if (algorithm_opt not in ["lsa", "greedy"]):
            raise ValueError("Algorithm_rd parameter must be in [lsa,greedy]")

        if (group_method not in ["avg", "rev"]):
            raise ValueError("Group method must be in ['avg', 'rev']")

        super().__init__(problem=problem, verbose=verbose, transformer=transformer)
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
        # Mapping features coordinates with image
        self._mapping_features()

    def _mapping_features(self):
        assigner = AssignerFactory.get_assigner(self._assignment_method,
                                                algorithm=self._algorithm_opt)
        self._features_positions = assigner.assign(self._features_coord, self._image_dim)
        self._features_relevance = self._compute_relevance(self._features_coord, self._features_positions)
        if (self._relocate):
            if (self._duplicated(self._features_positions)):
                optimizer = AssignerFactory.get_assigner('RelevanceAssigner', algorithm=self._algorithm_opt)
                self._features_positions = optimizer.assign(self._features_relevance, self._image_dim)


    def _duplicated(self, features_positions):
        df_feature_pos = pd.DataFrame(features_positions)
        return df_feature_pos.duplicated().any()

    def _compute_relevance(self, features_coord, features_positions):
        norm = np.linalg.norm(features_coord, axis=1).reshape(-1, 1) + 1
        return np.hstack((features_positions, norm))

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        imgs_coord = self._calculate_pixels_values(x)
        # Create and Save Images
        self._create_images(imgs_coord, y)
        self._features_pos_to_csv(x.columns,self._features_positions)


    def _calculate_pixels_values(self,x):
        """
        Calls the specified method for calculating the value of each pixel.
        Args:
            x: Dataset

        Returns:

        """
        match self._group_method:
            case 'avg':
                return self._avg_features_by_pixels(x)
            case 'rev':
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
        features_dot_rev=(x.T)*(self._features_relevance[:,2].reshape(-1,1))
        rev_sum = (pd.DataFrame(self._features_relevance).groupby([0, 1], as_index=False).transform('sum')).to_numpy()
        features_rev=features_dot_rev/rev_sum
        pixels_sum=(pd.DataFrame(np.hstack((self._features_positions, features_rev))).groupby([0, 1], as_index=False).sum()).to_numpy()
        rev_count = (pd.DataFrame(self._features_relevance).groupby([0, 1], as_index=False).count()).to_numpy()
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


    def _img_to_file(self,image,file,extension):
        matplotlib.image.imsave(file, image, format=extension,cmap='gray',vmin=0,vmax=1)

    def get_dim(self):
        return self._image_dim

    @abstractmethod
    def _get_features_coords(self,x):
        raise NotImplementedError("Subclasses must implement _fit_alg.")
