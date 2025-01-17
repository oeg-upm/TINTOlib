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

# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod
from TINTOlib.utils import Toolbox

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
        Default is 'supervised'. Valid values: ['supervised', 'unsupervised', 'regression'].
    verbose : bool, optional
        Show execution details in the terminal. 
        Default is False. Valid values: [True, False].
    hcIterations : int, optional
        Number of iterations for the hill climbing algorithm. 
        Default is 5. Valid values: integer >= 1.
    random_seed : int, optional
        Seed for reproducibility. 
        Default is 1. Valid values: integer.
    zoom : int, optional
        Multiplication factor determining the size of the saved image relative to the original size. 
        Default is 1. Valid values: integer > 0.
    n_processors : int, optional    
        The number of processors to use for the algorithm. 
        Default is 8. Valid values: integer >= 2.
    """
    ###### default values ###############
    default_hc_iterations = 5  # Number of iterations for the hill climbing algorithm
    default_random_seed = 1  # Default seed for reproducibility
    default_zoom = 1  # Default zoom level for saving images
    default_n_processors = 8  # Default number of processors

    def __init__(
        self,
        problem: Optional[str] = None,
        verbose: Optional[bool] = None,
        hcIterations: Optional[int] = default_hc_iterations,
        random_seed: Optional[int] = default_random_seed,
        zoom: Optional[int] = default_zoom,
        n_processors: Optional[int] = default_n_processors
    ):   
        super().__init__(problem=problem, verbose=verbose)
        if n_processors < 2:
            raise ValueError(f"n_processors must be greater than 1 (got {n_processors})")
        
        self.hcIterations = hcIterations
        self.random_seed = random_seed
        self.zoom = zoom
        self.n_processors = n_processors

    def __saveSupervised(self, classValue, i, folder, matrix_a, fig, ax):
        extension = 'png'  # eps o pdf
        subfolder = str(int(classValue)).zfill(2)  # subfolder for grouping the results of each class
        name_image = str(i).zfill(6)
        route = os.path.join(folder, subfolder)
        route_complete = os.path.join(route, name_image + '.' + extension)
        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")

        shape = int(math.sqrt(matrix_a.shape[0]))
        data = matrix_a.reshape(shape, shape)

        fig.set_size_inches(shape, shape)
        fig.set_dpi(self.zoom)
        ax.clear()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(data, cmap='viridis', interpolation="nearest")
        ax.axis('off')
        fig.canvas.draw()
        fig.savefig(fname=route_complete, pad_inches=0, bbox_inches='tight', dpi=self.zoom)
        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __saveRegressionOrUnsupervised(self, i, folder, matrix_a, fig, ax):
        extension = 'png'  # eps o pdf
        subfolder = "images"
        name_image = str(i).zfill(6)  + '.' + extension
        route = os.path.join(folder, subfolder)
        route_complete = os.path.join(route, name_image)

        if not os.path.isdir(route):
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")

        shape = int(math.sqrt(matrix_a.shape[0]))
        data = matrix_a.reshape(shape,shape)

        fig.set_size_inches(shape, shape)
        fig.set_dpi(self.zoom)
        ax.clear()
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(data, cmap='viridis', interpolation="nearest")
        ax.axis('off')
        fig.canvas.draw()
        fig.savefig(fname=route_complete, pad_inches=0, bbox_inches='tight', dpi=self.zoom)
        route_relative = os.path.join(subfolder, name_image)
        return route_relative

    def __saveImages(self,gene_names,coords,map_in_int, X, Y, nn):

        gene_names_MDS, coords_MDS, map_in_int_MDS=(gene_names,coords,map_in_int)
        X_REFINED_MDS = Toolbox.REFINED_Im_Gen(X, nn, map_in_int_MDS, gene_names_MDS, coords_MDS)
        imagesRoutesArr=[]
        total = Y.shape[0]

        if self.verbose:
            print("SAVING")
        
        fig,ax = plt.subplots()

        for i in range(len(X_REFINED_MDS)):
            if self.problem == "supervised":
                route=self.__saveSupervised(Y[i], i, self.folder, X_REFINED_MDS[i], fig, ax)
                imagesRoutesArr.append(route)

            elif self.problem == "unsupervised" or self.problem == "regression" :
                route = self.__saveRegressionOrUnsupervised(i, self.folder, X_REFINED_MDS[i], fig, ax)
                imagesRoutesArr.append(route)
            else:
                print("Wrong problem definition. Please use 'supervised', 'unsupervised' or 'regression'")
            if self.verbose:
                print("Created ", str(i+1), "/", int(total))

        if self.problem == "supervised" :
            data={'images':imagesRoutesArr,'class':Y}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/supervised.csv", index=False)
        elif self.problem == "unsupervised":
            data = {'images': imagesRoutesArr}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/unsupervised.csv", index=False)
        elif self.problem == "regression":
            data = {'images': imagesRoutesArr,'values':Y}
            regressionCSV = pd.DataFrame(data=data)
            regressionCSV.to_csv(self.folder + "/regression.csv", index=False)

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
            command = f'mpiexec --use-hwthread-cpus -np {self.n_processors} python {script_path} --init "{init_pickle_file}" --mapping "{mapping_pickle_file}"  --evolution "{evolution_csv_file}" --num {self.hcIterations}'
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