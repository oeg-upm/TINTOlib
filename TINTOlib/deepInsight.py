import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from TINTOlib.utils.geometry import get_minimum_rectangle
from TINTOlib.paramImageMethod import ParamImageMethod
from sklearn.preprocessing import MinMaxScaler

###########################################################
################    DeepInsight    ##############################
###########################################################


default_algorithm_rd = 'PCA'
default_assignment_method = 'bin'
default_random_seed = 23,
default_algorithm_opt = 'lsa'
default_group_method = 'avg'
default_zoom=1
default_format = 'png'  # Default output format
default_cmap = 'gray'  # Default cmap image output


class DeepInsight(ParamImageMethod):
    """
    DeepInsight: A class for transforming tabular data into synthetic images by projecting it into a two-dimensional space
    using DeepInsight method. The original implementation github repository is https://github.com/alok-ai-lab/pyDeepInsight/tree/master.
    We allow using a different bin discretization method and using an optimizer to resolve pixels shared based on relevance when PCA its selected as features coordinates
    extractor method.

    Parameters:
    ----------
    image_dim : int
        Order size for a square matrix image
    problem : str, optional
        The type of problem, defining how the images are grouped.
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
   transformer : CustomTransformer, optional
        Preprocessing transformations like scaling, normalization,etc.
        Default is None
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


    def __init__(
            self,
            image_dim,
            problem=None,
            transformer=MinMaxScaler(),
            verbose=None,
            algorithm_rd=default_algorithm_rd,
            assignment_method=default_assignment_method,
            relocate=False,
            algorithm_opt=default_algorithm_opt,
            group_method=default_group_method,
            zoom=default_zoom,
            format=default_format,
            cmap=default_cmap,
            random_seed=default_random_seed
    ):
        if (algorithm_rd not in ["PCA", "t-SNE", 'KPCA']):
            raise TypeError("Algorithm_rd parameter must be in [PCA,t-SNE,KPCA]")

        if(algorithm_rd != "PCA"): relocate = False

        super().__init__( image_dim,problem,transformer,verbose,assignment_method,relocate,algorithm_opt,group_method,zoom,format,cmap,random_seed)

        self.__algorithm_rd = algorithm_rd



    def __apply_dimensionality_reduction(self,x):
        """
        Apply a dimensionality reduction technique to get features coordinates
        Args:
            x: features tabular data

        Returns:
            features coordinate matrix
        """
        match self.__algorithm_rd:
            case 'PCA':
                return PCA(n_components=2,random_state=self._random_seed).fit_transform(x)
            case 't-SNE':
                return TSNE(n_components=2,metric='cosine',random_state=self._random_seed,perplexity=self._image_dim).fit_transform(x)
            case 'KPCA':
                return KernelPCA(n_components=2,kernel='rbf',random_state=self._random_seed).fit_transform(x)


    def _get_features_coords(self, x):
        """
                This method computes the features coordinates matrix
                Args:
                    x: features tabular data

                Returns:
                    features coordinate matrix
        """
        features_coord = self.__apply_dimensionality_reduction(x)
        # Apply Convex Hull algorithm to get limit points, calculate mimimun rectangle area and rotate
        rotmat, rect_coords, limit_points = get_minimum_rectangle(features_coord)
        return np.dot(rotmat, features_coord.T).T