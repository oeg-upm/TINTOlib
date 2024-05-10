import os
import matplotlib.pyplot as plt
import subprocess
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
import math
import pickle
import numpy as np
import pandas as pd
import pickle
from TINTOlib import utils
from TINTOlib.utils.Toolbox import two_d_eq, Assign_features_to_pixels,REFINED_Im_Gen
import platform
import os

class REFINED:
    default_problem = "supervised"  # Define the type of dataset [supervised, unsupervised, regression]
    default_verbose = False     # Verbose: if it's true, show the compilation text
    default_hc_iterations = 5   #Number of iterations is basically how many times the hill climbing goes over the entire features and check each feature exchange cost
    default_random_seed = 1     # Default seed to generate the embeddings using MDS
    default_scale_up = True
    def __init__(
            self,
            problem=default_problem,
            verbose=default_verbose,
            hcIterations=default_hc_iterations,
            random_seed=default_random_seed,
            scale_up = default_scale_up
        ):
        self.verbose = verbose
        self.problem = problem
        self.hcIterations = hcIterations
        self.random_seed = random_seed
        self.scale_up = scale_up # TODO: implement

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

        if self.verbose:
            print("It has been successfully loaded in " + filename)


    def __imageSampleFilter(self, X, Y, coord, matrix, folder):
        """
        This function creates the samples, i.e., the images. This function has the following specifications:
        - The first conditional performs the pre-processing of the images by creating the matrices.
        - Then the for loop generates the images for each sample. Some assumptions have to be taken into
          account in this step:
            - The samples will be created according to the number of targets. Therefore, each folder that is
              created will contain the images created for each target.
            - In the code, the images are exported in PNG format; this can be changed to any other format.
        """


    def scatter_list_to_processors(self,comm, data_list, n_processors):
        import math
        data_amount = len(data_list)
        heap_size = math.ceil(data_amount / (n_processors))

        for pidx in range(1, n_processors):
            try:
                heap = data_list[heap_size * (pidx - 1):heap_size * pidx]
            except:
                heap = data_list[heap_size * (pidx - 1):]
            comm.send(heap, dest=pidx)

        return True

    def receive_from_processors_to_dict(self,comm, n_processors):
        # receives dicts, combine them and return
        feedback = dict()
        for pidx in range(1, n_processors):
            receved = comm.recv(source=pidx)
            feedback.update(receved)
        return feedback

    def __createImage(self, X, Y, folder='prueba/', train_m=False):
        """
        This function creates the images that will be processed by CNN.
        """
        X_scaled = self.min_max_scaler.transform(X)
        Y = np.array(Y)
        try:
            os.mkdir(folder)
            if self.verbose:
                print("The folder was created " + folder + "...")
        except:
            if self.verbose:
                print("The folder " + folder + " is already created...")

        self.m = self.__imageSampleFilter(X_scaled, Y, self.pos_pixel_caract, self.m, folder)
    def __saveSupervised(self,classValue,i,folder,matrix_a):
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
        if self.scale_up:
            plt.imshow(matrix_a.reshape(shape, shape), cmap='viridis')
            plt.axis('off')
            plt.savefig(fname=route_complete, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.imsave(route_complete, matrix_a.reshape(shape,shape), cmap='viridis')
        route_relative = os.path.join(subfolder, name_image+ '.' + extension)
        return route_relative

    def __saveRegressionOrUnsupervised(self, i, folder, matrix_a):
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
        if self.scale_up:
            plt.imshow(matrix_a.reshape(shape,shape), cmap='viridis')
            plt.axis('off')
            plt.savefig(fname=route_complete, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.imsave(route_complete, matrix_a.reshape(shape,shape), cmap='viridis')
        route_relative = os.path.join(subfolder, name_image)
        return route_relative

    def __saveImages(self,gene_names,coords,map_in_int, X, Y, nn):

        gene_names_MDS, coords_MDS, map_in_int_MDS=(gene_names,coords,map_in_int)
        X_REFINED_MDS = utils.Toolbox.REFINED_Im_Gen(X, nn, map_in_int_MDS, gene_names_MDS, coords_MDS)
        imagesRoutesArr=[]
        total = Y.shape[0]
        #print(X_REFINED_MDS.shape)
        print("SAVING")
        for i in range(len(X_REFINED_MDS)):
            if self.problem == "supervised":
                route=self.__saveSupervised(Y[i], i, self.folder, X_REFINED_MDS[i])
                imagesRoutesArr.append(route)

            elif self.problem == "unsupervised" or self.problem == "regression" :
                route = self.__saveRegressionOrUnsupervised(i, self.folder, X_REFINED_MDS[i])
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


    def __trainingAlg(self, X, Y,Desc):
        """
        This function uses the above functions for the training.
        """
        #Feat_DF = pd.read_csv(data)

        #X = Feat_DF.values;

        #X = X[:100, :-1]


        original_input = pd.DataFrame(data=X)  # The MDS input should be in a dataframe format with rows as samples and columns as features
        feature_names_list = original_input.columns.tolist()  # Extracting feature_names_list (gene_names or descriptor_names)
        if self.verbose:
            print(">>>> Data  is loaded")

        # %% MDS
        nn = math.ceil(np.sqrt(len(feature_names_list)))  # Image dimension
        Nn = original_input.shape[1]  # Number of features

        transposed_input = original_input.T  # The MDS input data must be transposed , because we want summarize each feature by two values (as compard to regular dimensionality reduction each sample will be described by two values)
        Euc_Dist = euclidean_distances(transposed_input)  # Euclidean distance
        Euc_Dist = np.maximum(Euc_Dist, Euc_Dist.transpose())  # Making the Euclidean distance matrix symmetric

        embedding = MDS(n_components=2, random_state=self.random_seed)  # Reduce the dimensionality by MDS into 2 components
        mds_xy = embedding.fit_transform(transposed_input)  # Apply MDS

        if self.verbose:
            print(">>>> MDS dimensionality reduction is done")

        eq_xy = utils.Toolbox.two_d_eq(mds_xy, Nn)
        Img = utils.Toolbox.Assign_features_to_pixels(eq_xy, nn,verbose=self.verbose)  # Img is the none-overlapping coordinates generated by MDS

        Desc = original_input.columns.tolist()                              # Drug descriptors name
        Dist = pd.DataFrame(data = Euc_Dist, columns = Desc, index = Desc)	# Generating a distance matrix which includes the Euclidean distance between each and every descriptor
        data = (Desc, Dist, Img	)  											# Preparing the hill climbing inputs

        init_pickle_file = "Init_MDS_Euc.pickle"
        with open(init_pickle_file, 'wb') as f:					# The hill climbing input is a pickle, therefore everything is saved as a pickle to be loaded by the hill climbing
            pickle.dump(data, f)

        mapping_pickle_file = "Mapping_REFINED_subprocess.pickle"
        evolution_csv_file = "REFINED_Evolve_subprocess.csv"
        script_path = os.path.join("utils", "mpiHill_UF.py")
        
        if 'Windows' == platform.system():
            command = f'mpiexec -np 40 python {script_path} --init "{init_pickle_file}" --mapping "{mapping_pickle_file}"  --evolution "{evolution_csv_file}" --num {self.hcIterations}'
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
        else:
            command = f"mpirun -np 40 python3 -u {script_path} --init {init_pickle_file} --mapping 'Mapping_REFINED.pickle'  --evolution {mapping_pickle_file}.csv --num {self.hcIterations}"
            result = subprocess.run(command, shell=True, text=True, capture_output=True)

        if result.returncode != 0:
            raise Exception(result.stderr)

        with open(mapping_pickle_file,'rb') as file:
            gene_names_MDS,coords_MDS,map_in_int_MDS = pickle.load(file)

        self.__saveImages(gene_names_MDS, coords_MDS, map_in_int_MDS, X, Y, nn)

        os.remove(init_pickle_file)
        os.remove(mapping_pickle_file)
        os.remove(evolution_csv_file)

    def generateImages(self,data, folder="/refinedData"):
        """
            This function generate and save the synthetic images in folders.
                - data : data CSV or pandas Dataframe
                - folder : the folder where the images are created
        """
        # Blurring verification

        # Read the CSV
        self.folder = folder
        if type(data) == str:
            dataset = pd.read_csv(data)
            array = dataset.values
            Desc=dataset.columns[:-1].tolist()
        elif isinstance(data, pd.DataFrame):
            array = data.values
            Desc = data.columns[:-1].tolist()

        X = array[:, :-1]
        Y = array[:, -1]

        # Training
        self.__trainingAlg(X, Y,Desc)
        if self.verbose: print("End")