import numpy as np
import pandas as pd  
import os
import gc

# Dimensional reduction classes 
from sklearn.manifold import TSNE
#from tsnecuda import TSNE
from sklearn.decomposition import PCA

#Sklearn
from sklearn.preprocessing import MinMaxScaler

# Graphic library
import matplotlib 
import matplotlib.image

# Additional libraries
import math
import pickle

# Arguments Library
import argparse

##################
#Params
parser = argparse.ArgumentParser(description="This program transform tidy data "+
                                 "into image by dimensionality "+
                                 "reduction algorithms (PCA o t-SNE)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-alg", "--algorithm", dest="algorithm", default="PCA", choices=['PCA','t-SNE'], help="dimensionality reduction algorithm (PCA o t-SNE)")
parser.add_argument("-px",  "--pixels", dest="pixels", default=20, help="Image's Pixels (one side)", type=int)


parser.add_argument("-B",  "--blurr", dest="blurr_active", action='store_true', help="Active option blurring")
parser.add_argument("-aB",  "--amplification_blurr", dest="amplification", default=np.pi, help="Amplification in blurring", type=float)
parser.add_argument("-dB",  "--distance_blurr", dest="distance", default=0.1, help="Distance in blurring (porcentage 0 to 1)", type=float)
parser.add_argument("-sB",  "--steps_blurr", dest="steps", default=4, help="Steps in blurring", type=int)
parser.add_argument("-oB",  "--option_blurr", dest="option", default='mean', choices=['mean','maximum'], help="Option in blurring (mean and maximum)")

parser.add_argument("-sC",  "--save", dest="save_configuration", help="Save configurations (to reuse)")
parser.add_argument("-lC",  "--load", dest="load_configuration", help="Load configurations (.pkl)")

parser.add_argument("-sd",  "--seed", dest="seed", default=20, help="seed", type=int)
parser.add_argument("-tt",  "--times_tsne", dest="times_tsne", default=4, help="Times replication in t-SNE", type=int)

parser.add_argument("src_data", help="Source location (tidy data in csv without head)")
parser.add_argument("dest_folder", help="Destination location (folder)")

parser.add_argument("-v",  "--verbose", dest="verbose", action='store_true', help="Verbose: if it's true, show the compilation text")
args = parser.parse_args()

###########################################################
################ TINTO MAIN FUNCTIONS  ####################
###########################################################

def square(coord):
    """
    This functionhas the purpose of being able to create the square delimitation of the resulting image. 
    The steps it performs in the order of the code are as follows: 
        - Calculate the average of the points $(x,y)$.
        - Centres the data at the point $(0,0)$.
        - Calculate the distance from the centre to the points.
        - The larger distance of \texttt{dista}, transforms it to integer.
        - Calculate the vertices of the square.
        - Move the points to quadrant $4$.
        - Transfers the vertices as well.
        - Returns the values, coordinates, and vertices.
    """
    m = np.mean(coord,axis=0).reshape((1,2))       
    coord_new = coord - m                        
    dista = (coord_new[:,0]**2+coord_new[:,1]**2)**0.5 
    maxi = math.ceil(max(dista))                   
    vertices = np.array([[-maxi,maxi],[-maxi,-maxi],[maxi,-maxi],[maxi,maxi]])  
    coord_new = coord_new - vertices[0]   
    vertices = vertices - vertices[0]
    return coord_new,vertices


def m_imagen(coord,vertices,filename,pixeles=24):
    """
    This function obtain the coordinates of the matrix. This function has 
    the following specifications:
        - Create a matrix of coordinates and vertices.
        - Transform the coordinates into indices for the matrix.
        - Fill in the positions of the features.
        - Finally, a conditional is created if the features were grouped 
          in the same position.
    """
    size = (pixeles,pixeles)
    matriz = np.zeros(size)
    
    coord_m = (coord/vertices[2,0])*(pixeles-1)
    coord_m = np.round(abs(coord_m))
    
    for i,j in zip(coord_m[:,1],coord_m[:,0]):
        matriz[int(i),int(j)] = 1       
    
    if(np.count_nonzero(matriz!=0)!=coord.shape[0]):
        return coord_m, matriz, True
    else:
        return coord_m, matriz, False


def blurring(matriz, coordinate, distance=0.1, steps=3, amplification=np.pi, option='maximum'):
    """
   This function is to be able to add more ordered contextual information to the image through the
   classical painting technique called blurring. This function develops the following main steps:
   - Take the coordinate matrix of the characteristic pixels.
   - Create the blurring according to the number of steps taken in a loop with the 
     following specifications:
        - Take the current radius (from largest to smallest).
        - Take the intensity of each step it makes
        - Delimit the blurring area according to $(x,y)$ on an upper and lower boundary.
        - Set the new intensity values in the matrix, taking into account that if there is 
          pixel overlap, the maximum or average will be taken as specified.
    """
    x = int(coordinate[1])
    y = int(coordinate[0])
    core_value = matriz[x,y]
    
    for p in range(steps):
        r_actual = int(matriz.shape[0]*distance*(p+1))   
        intensity=min(amplification*core_value/(np.pi*r_actual**2),core_value)
        
        # Delimitation of the area
        lim_inf_i = max(x-r_actual-1,0)
        lim_sup_i = min(x+r_actual+1,matriz.shape[0])
        lim_inf_j = max(y-r_actual-1,0)
        lim_sup_j = min(y+r_actual+1,matriz.shape[1])
        
        for i in range(lim_inf_i, lim_sup_i):
            for j in range(lim_inf_j, lim_sup_j):
                if((x-i)**2 + (y-j)**2 <= r_actual**2):
                    if(matriz[i,j]==0):
                        matriz[i,j]=intensity
                    elif(x!=i and y!=j): # Pixel overlapping
                        if(option=='mean'):
                            matriz[i,j]=(matriz[i,j]+intensity)/2 
                        elif(option=='maximum'):
                            matriz[i,j]=max(matriz[i,j],intensity)
    return matriz

   
def imageSample(X, Y, coord, matriz, folder, amplification, distance=0.1, steps=3, option='maximum', train_m=False):
    """
    This function creates the samples, i.e., the images. This function has the following specifications:
    - The first conditional performs the pre-processing of the images by creating the matrices.
    - Then the for loop generates the images for each sample. Some assumptions have to be taken into 
      account in this step:
        - The samples will be created according to the number of targets. Therefore, each folder that is 
          created will contain the images created for each target. 
        - In the code, the images are exported in PNG format; this can be changed to any other format.
    """

    if train_m:
        matriz_a = np.zeros(matriz.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
                matriz_a = blurring(matriz_a, coord[j], distance, steps, amplification, option)
        matriz = np.copy(matriz_a)
    
    # In this part, images are generated for each sample.
    for i in range(X.shape[0]):
        matriz_a = np.copy(matriz)
        
        for j in range(X.shape[1]):
            matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
            matriz_a = blurring(matriz_a, coord[j], distance, steps, amplification, option)
        
        for j in range(X.shape[1]):
            matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
        
        extension = 'png'   # eps o pdf
        subfolder = str(int(Y[i])).zfill(2)    # sub-folder to group the results of each class
        image_name = str(i).zfill(6)
        path = os.path.join(folder, subfolder)
        path_full = os.path.join(path, image_name+'.'+extension)
        if not os.path.isdir(path):            
            try:
                os.makedirs(path)
            except:
                print("Error: Could not create subfolder")
        matplotlib.image.imsave(path_full, matriz_a, cmap='binary', format=extension)
                
    return matriz

def saveVariable(X, filename='objs.pkl',verbose=False):
    """
    This function allows SAVING the transformation options to images in a Pickle object. 
    This point is basically to be able to reproduce the experiments or reuse the transformation 
    on unlabelled data.
    """
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
    if(verbose):
        print("It has been successfully saved in "+filename)

def loadVariable(filename='objs.pkl',verbose=False):
    """
    This function allows LOADING the transformation options to images in a Pickle object. 
    This point is basically to be able to reproduce the experiments or reuse the transformation 
    on unlabelled data.
    """
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    if(verbose):
        print("It has been successfully loaded in "+filename)
    return variable


###########################################################
################   TINTO MAIN CLASS    ####################
###########################################################

class DataImg:
    """
    Python class has been developed that contains different specific functions 
    related to each step in the data transformation process
    """
    
    def __init__(self, algorithm='PCA', pixeles=20, seed=20, times=4, amp=np.pi, distance=0.1, steps=4, option='maximum'):
        """
        This function initialises packages and objects in Python, i.e., displays 
        the initialisation of each object.
        """
        self.algorithm = algorithm  # Dimensional reduction algorithm
        self.p = pixeles            
        self.seed = seed            
        self.times = times  # only for t-sne
        
        self.amp = amp              # amplitude (blurring)
        self.distance = distance    # distance (blurring)
        self.steps = steps          # steps (blurring)
        self.option = option        # overlapping option (blurring)
        
        self.error_pos = False      # Indicates the overlap of characteristic pixels.
        
    def obtainCoord(self, X, verbose=False):
        """
        This function uses the dimensionality reduction algorithm in order to represent the characteristic 
        pixels in the image. The specifications of this function are:
        - Perform a normalisation of (0,1) to be able to represent the pixels inside the square. 
        - Transpose the matrix.
        - Set the dimensionality reduction algorithm, PCA or t-SNE. 
        """

        self.min_max_scaler = MinMaxScaler()
        X = self.min_max_scaler.fit_transform(X)

        labels = np.arange(X.shape[1])
        X_trans = X.T
        
        if(verbose):
            print("Selected algorithm: "+self.algorithm)
            
        if(self.algorithm=='PCA'):
            X_embedded = PCA(n_components=2,random_state=self.seed).fit(X_trans).transform(X_trans)
        elif(self.algorithm=='t-SNE'):
            for i in range(self.times):
                X_trans = np.append(X_trans,X_trans,axis=0)
                labels = np.append(labels,labels,axis=0)
            X_embedded = TSNE(n_components=2,random_state=self.seed,perplexity=50).fit_transform(X_trans)
        else:
            print("Error: Incorrect algorithm")
            X_embedded = np.random.rand(X.shape[1],2)
        
        data_coord = {'x':X_embedded[:,0], 'y':X_embedded[:,1], 'Label':labels}
        dc = pd.DataFrame(data=data_coord)
        self.obtain_coord = dc.groupby('Label').mean().values

        del X_trans
        gc.collect()
    
    def areaDelimitation(self):
        """
        This function performs the delimitation of the area
        """
        self.initial_coordinates, self.vertices = square(self.obtain_coord)
        
    def matrixPositions(self, filename='original'):
        """
        This function gets the positions in the matrix
        """
        self.pos_pixel_caract, self.m, self.error_pos = m_imagen(self.initial_coordinates,self.vertices,filename,pixeles=self.p)
        
    def CrearImg(self, X, Y, folder = 'prueba/', train_m=False, verbose=False):
        """
        This function creates the images that will be processed by CNN.
        """
        
        X_scaled = self.min_max_scaler.transform(X)
        Y = np.array(Y)
        try:
            os.mkdir(folder)
            if(verbose):
                print("The folder was created "+folder+"...")
        except:
            if(verbose):
                print("The folder "+folder+" is already created...")
         
        self.m = imageSample(X_scaled, Y, self.pos_pixel_caract, self.m, folder, self.amp, distance=self.distance, steps=self.steps, option=self.option, train_m=train_m)
        
    def trainingAlg(self, X, Y, folder = 'img_train/', verbose=False):
        """
        This function uses the above functions for the training.
        """
        self.obtainCoord(X, verbose=verbose)
        self.areaDelimitation()
        self.matrixPositions()
        self.CrearImg(X, Y, folder, train_m=True, verbose=verbose)
        
    def testAlg(self, X, Y=None, folder = 'img_test/', verbose=False):
        """
        This function uses the above functions for the validation.
        """
        if(Y is None):
            Y = np.zeros(X.shape[0])
        self.CrearImg(X, Y, folder, train_m=False, verbose=verbose) 


###########################################################
################    TINTO EXECUTION    ####################
###########################################################

# Blurring verification
if not args.blurr_active:
    args.amplification = 0
    args.distance = 0.1
    args.steps = 0


# Read the CSV
dataset = pd.read_csv(args.src_data)
array = dataset.values
if args.load_configuration:
    X = array
    modeloIMG = loadVariable(filename=args.load_configuration,verbose=args.verbose)
    
    modeloIMG.testAlg(X,folder=args.dest_folder,verbose=args.verbose)
else:
    X = array[:,:-1]
    Y = array[:,-1]
    # Create the object
    modeloIMG = DataImg(algorithm=args.algorithm,
                        pixeles=args.pixels,
                        amp=args.amplification,
                        distance=args.distance,
                        steps=args.steps,
                        option=args.option,
                        seed=args.seed,
                        times=args.times_tsne
                        )

    # Training
    modeloIMG.trainingAlg(X, Y, folder=args.dest_folder, verbose=args.verbose)

# Saves the configuration for later use
if args.save_configuration:
    saveVariable(modeloIMG, filename=args.save_configuration,verbose=args.verbose)
