# Standard library imports
import math
import os
import pickle
import platform
import subprocess

# Third-party library imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances

# Typing imports
from typing import Optional, Union

from sklearn.preprocessing import MinMaxScaler

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod
from TINTOlib.utils import Toolbox, CustomTransformer


###########################################################
################    REFINED    ##############################
###########################################################

class REFINED(AbstractImageMethod):
    """
    REFINED: A method to transform high-dimensional feature vectors into 2D images 
    optimized for CNNs. It uses Bayesian Metric Multidimensional Scaling (BMDS) 
    and hill climbing to arrange features spatially, preserving neighborhood dependencies.
    
    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
    transformer : CustomTransformer, optional
        Preprocessing transformations like scaling, normalization,etc.
        Default is MinMaxScaler.
        Valid: Scikit Learn transformers or custom implementation using inheritance over CustomTransformer class.xº
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    hcIterations : int, optional
        Number of iterations for the hill climbing algorithm. 
        Default is 5. Valid values: integer >= 1.
    n_processors : int, optional    
        The number of processors to use for the algorithm. 
        Default is 8. Valid values: integer >= 2.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    random_seed : int, optional
        Seed for reproducibility. 
        Default is 1. Valid values: integer.
    """
    ###### default values ###############
    default_hc_iterations = 5  # Number of iterations for the hill climbing algorithm
    default_n_processors = 8  # Default number of processors

    default_zoom = 1  # Default zoom level for saving images
    default_random_seed = 1  # Default seed for reproducibility

    def __init__(
        self,
        problem: Optional[str] = None,
        transformer: Optional[CustomTransformer] = MinMaxScaler(),
        verbose: Optional[bool] = None,
        hcIterations: Optional[int] = default_hc_iterations,
        n_processors: Optional[int] = default_n_processors,
        zoom: Optional[int] = default_zoom,
        random_seed: Optional[int] = default_random_seed,
    ):   
        super().__init__(problem=problem, verbose=verbose, transformer=transformer)
        if n_processors < 2:
            raise ValueError(f"n_processors must be greater than 1 (got {n_processors})")
        
        self.hcIterations = hcIterations
        self.n_processors = n_processors

        self.zoom = zoom
        self.random_seed = random_seed

    def _img_to_file(self,image_matrix,file,extension):
        shape = int(math.sqrt(image_matrix.shape[0]))
        data = image_matrix.reshape(shape, shape)
        plt.axis('off')
        plt.imshow(data, cmap='viridis', interpolation="nearest")
        plt.savefig(file,pad_inches=0, bbox_inches='tight', dpi=self.zoom)
        
    def __saveImages(self,gene_names,coords,map_in_int, X, Y, nn):

        gene_names_MDS, coords_MDS, map_in_int_MDS=(gene_names,coords,map_in_int)
        X_REFINED_MDS = Toolbox.REFINED_Im_Gen(X, nn, map_in_int_MDS, gene_names_MDS, coords_MDS)

        if self.verbose:
            print("SAVING")

        for i in range(len(X_REFINED_MDS)):
            self._save_image(X_REFINED_MDS[i],Y[i],i)

    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        Desc = x.columns.tolist()
        X = x.values
        Y = y.values if y is not None else None

        original_input = pd.DataFrame(data=X)  # The MDS input should be in a dataframe format with rows as samples and columns as features
        feature_names_list = original_input.columns.tolist()  # Extracting feature_names_list (gene_names or descriptor_names)
        if self.verbose:
            print(">>>> Data  is loaded")

        nn = math.ceil(np.sqrt(len(feature_names_list)))  # Image dimension
        Nn = original_input.shape[1]  # Number of features

        transposed_input = original_input.T  # The MDS input data must be transposed , because we want summarize each feature by two values (as compard to regular dimensionality reduction each sample will be described by two values)
        Euc_Dist = euclidean_distances(transposed_input)  # Euclidean distance
        Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())  # Making the Euclidean distance matrix symmetric

        embedding = MDS(n_components=2, random_state=self.random_seed)  # Reduce the dimensionality by MDS into 2 components
        mds_xy = embedding.fit_transform(transposed_input)  # Apply MDS

        if self.verbose:
            print(">>>> MDS dimensionality reduction is done")

        eq_xy = Toolbox.two_d_eq(mds_xy, Nn)
        Img = Toolbox.Assign_features_to_pixels(eq_xy, nn,verbose=self.verbose)  # Img is the none-overlapping coordinates generated by MDS

        Desc = original_input.columns.tolist()                              # Drug descriptors name
        Dist = pd.DataFrame(data = Euc_Dist, columns = Desc, index = Desc)	# Generating a distance matrix which includes the Euclidean distance between each and every descriptor
        data = (Desc, Dist, Img	)  											# Preparing the hill climbing inputs

        init_pickle_file = "Init_MDS_Euc.pickle"
        with open(init_pickle_file, 'wb') as f:					# The hill climbing input is a pickle, therefore everything is saved as a pickle to be loaded by the hill climbing
            pickle.dump(data, f)

        mapping_pickle_file = "Mapping_REFINED_subprocess.pickle"
        evolution_csv_file = "REFINED_Evolve_subprocess.csv"
        script_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "utils","mpiHill_UF.py"
        )
        
        if 'Windows' == platform.system():
            command = f'mpiexec -np {self.n_processors} python {script_path} --init "{init_pickle_file}" --mapping "{mapping_pickle_file}"  --evolution "{evolution_csv_file}" --num {self.hcIterations}'
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
        else:
            command = f'mpirun --allow-run-as-root --use-hwthread-cpus -np {self.n_processors} python3 {script_path} --init "{init_pickle_file}" --mapping "{mapping_pickle_file}"  --evolution "{evolution_csv_file}" --num {self.hcIterations}'
            result = subprocess.run(command, shell=True, text=True, capture_output=True)

        if result.returncode != 0:
            raise Exception(result.stderr)

        with open(mapping_pickle_file,'rb') as file:
            self.gene_names_MDS, self.coords_MDS, self.map_in_int_MDS = pickle.load(file)
    
        os.remove(init_pickle_file)
        os.remove(mapping_pickle_file)
        os.remove(evolution_csv_file)

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        X = x.values
        Y = y.values if y is not None else None

        original_input = pd.DataFrame(data=X)  # The MDS input should be in a dataframe format with rows as samples and columns as features
        feature_names_list = original_input.columns.tolist()  # Extracting feature_names_list (gene_names or descriptor_names)
        nn = math.ceil(np.sqrt(len(feature_names_list)))  # Image dimension
        self.__saveImages(self.gene_names_MDS, self.coords_MDS, self.map_in_int_MDS, X, Y, nn)