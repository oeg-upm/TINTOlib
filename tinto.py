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
parser.add_argument("-dB",  "--distance_blurr", dest="distance", default=2, help="Distance in blurring (number of pixels)", type=int)
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
    matrix = np.zeros(size)
    
    coord_m = (coord/vertices[2,0])*(pixeles-1)
    coord_m = np.round(abs(coord_m))
    
    for i,j in zip(coord_m[:,1],coord_m[:,0]):
        matrix[int(i),int(j)] = 1       
    
    if(np.count_nonzero(matrix!=0)!=coord.shape[0]):
        return coord_m, matrix, True
    else:
        return coord_m, matrix, False

def createFilter(distance=2, steps=3, amplification=np.pi):
    """
    In this function a filter is created since a matrix of size "2*distance*total_steps+1" 
    is being created to act as a "filter", which covers the whole circular space of the minutiae 
    determined by the distance and by the total number of steps. 
    This "filter", which is a matrix, would be multiplied with a scalar, which is the intensity value. 
    Finally, this resulting matrix is placed as a submatrix within the final matrix where the centre 
    of the submatrix would be the position of the characteristic pixel.
    """
    size_filter = int(2 * distance * steps + 1)
    center_x = distance * steps
    center_y = distance * steps
    print(distance,steps)
    filter  = np.zeros([size_filter,size_filter])
    
    for step in reversed(range(steps)):
        r_actual = int(distance*(step+1))   # current radius from largest to smallest
        
        #Function of intensity
        intensity=min(amplification*1/(np.pi*r_actual**2),1)
        
        #Delimitation of the area
        lim_inf_i = max(center_x - r_actual - 1, 0)
        lim_sup_i = min(center_x + r_actual + 1, size_filter)
        lim_inf_j = max(center_y - r_actual - 1, 0)
        lim_sup_j = min(center_y + r_actual + 1, size_filter)
        
        #Allocation of values
        for i in range(lim_inf_i, lim_sup_i):
            for j in range(lim_inf_j, lim_sup_j):
                if((center_x-i)**2 + (center_y-j)**2 <= r_actual**2):
                    filter[i,j]=intensity
    filter[center_x,center_y] = 1
    return filter


def blurringFilter(matrix, filter, values, coordinates, option):
    """
   This function is to be able to add more ordered contextual information to the image through the
   classical painting technique called blurring. This function develops the following main steps:
   - Take the coordinate matrix of the characteristic pixels.
   - Create the blurring according to the number of steps taken in a loop with the 
     following specifications:
        - Delimit the blurring area according to $(x,y)$ on an upper and lower boundary.
        - Set the new intensity values in the matrix, taking into account that if there is 
          pixel overlap, the maximum or average will be taken as specified.
    """
    iter_values = iter(values)
    size_matrix = matrix.shape[0]
    size_filter = filter.shape[0]
    matrix_extended = np.zeros([size_filter+size_matrix,size_filter+size_matrix])
    matrix_add = np.zeros([size_filter+size_matrix,size_filter+size_matrix])
    center_filter = int((size_filter - 1)/2)
    for i,j in coordinates:
        i = int(i)
        j = int(j)
        value = next(iter_values)
        submatrix = filter * value

        #Delimitación del área
        lim_inf_i = i
        lim_sup_i = i+2*center_filter+1
        lim_inf_j = j
        lim_sup_j = j+2*center_filter+1

        if(option=='mean'):
            matrix_extended[lim_inf_i:lim_sup_i,lim_inf_j:lim_sup_j] += submatrix
            matrix_add[lim_inf_i:lim_sup_i,lim_inf_j:lim_sup_j] += (submatrix > 0)*1
        elif(option=='maximum'):
            matrix_extended[lim_inf_i:lim_sup_i,lim_inf_j:lim_sup_j] = np.maximum(matrix_extended[lim_inf_i:lim_sup_i,lim_inf_j:lim_sup_j], submatrix)

    if(option=='mean'):
        matrix_add[matrix_add == 0] = 1
        matrix_extended = matrix_extended / matrix_add

    matrix_final = matrix_extended[center_filter:-center_filter-1,center_filter:-center_filter-1]

    return matrix_final

def imageSampleFilter(X, Y, coord, matrix, folder, amplification, distance=2, steps=3, option='maximum', train_m=False):
    """
    This function creates the samples, i.e., the images. This function has the following specifications:
    - The first conditional performs the pre-processing of the images by creating the matrices.
    - Then the for loop generates the images for each sample. Some assumptions have to be taken into 
      account in this step:
        - The samples will be created according to the number of targets. Therefore, each folder that is 
          created will contain the images created for each target. 
        - In the code, the images are exported in PNG format; this can be changed to any other format.
    """
    
    # Generate the filter
    if distance * steps * amplification != 0:          # The function is only called if there are no zeros (blurring).
        filter = createFilter(distance,steps,amplification)
    
    # In this part, images are generated for each sample.
    for i in range(X.shape[0]):
        matrix_a = np.zeros(matrix.shape)
        if distance * steps * amplification != 0:      # The function is only called if there are no zeros (blurring).
            matrix_a = blurringFilter(matrix_a, filter, X[i], coord, option)
        else:   #(no blurring)
            iter_values_X = iter(X[i])
            for eje_x,eje_y in coord:
                matrix_a[int(eje_x),int(eje_y)]=next(iter_values_X)
        
        extension = 'png'   #eps o pdf
        subfolder = str(int(Y[i])).zfill(2)    # subfolder for grouping the results of each class
        name_image = str(i).zfill(6)
        route = os.path.join(folder,subfolder)
        route_complete = os.path.join(route,name_image+'.'+extension)
        if not os.path.isdir(route):            
            try:
                os.makedirs(route)
            except:
                print("Error: Could not create subfolder")
        matplotlib.image.imsave(route_complete, matrix_a, cmap='binary', format=extension)
        
    return matrix


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
    
    def __init__(self, algorithm='PCA', pixeles=20, seed=20, times=4, amp=np.pi, distance=2, steps=4, option='maximum'):
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
         
        self.m = imageSampleFilter(X_scaled, Y, self.pos_pixel_caract, self.m, folder, self.amp, 
                            distance=self.distance, steps=self.steps, option=self.option, train_m=train_m)
        
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
    args.distance = 2
    args.steps = 0


# Read the CSV
dataset = pd.read_csv(args.src_data)
array = dataset.values
if args.load_configuration:
    X = array
    modeloIMG = loadVariable(filename=args.load_configuration,verbose=args.verbose)
    
    modeloIMG.__testAlg(X, folder=args.dest_folder, verbose=args.verbose)
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
