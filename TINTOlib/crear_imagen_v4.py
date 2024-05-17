import numpy as np
import pandas as pd  
import os
import gc

# Funciones de reducción dimensional 
from sklearn.manifold import TSNE
#from tsnecuda import TSNE
from sklearn.decomposition import PCA

#Sklearn
from sklearn.preprocessing import MinMaxScaler

# Biblioteca grafica
import matplotlib 
import matplotlib.image

#Bibliotecas adicionales
import math
import pickle

#Para argumentos
import argparse

##################
#Parametros
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

parser.add_argument("-sd",  "--random_seed", dest="random_seed", default=20, help="seed for reproducibility", type=int)
parser.add_argument("-tt",  "--times_tsne", dest="times_tsne", default=4, help="Times replication in t-SNE", type=int)

parser.add_argument("src_data", help="Source location (tidy data in csv without head)")
parser.add_argument("dest_folder", help="Destination location (folder)")

parser.add_argument("-v",  "--verbose", dest="verbose", action='store_true', help="Verbose: if it's true, show some text")
args = parser.parse_args()


#Funciones 
#####
def cuadrado(coord):
    #Función de delimitación en forma de cuadrado.
    
    m = np.mean(coord,axis=0).reshape((1,2))       #promedio de los puntos (x,y)
    coord_nuevo = coord - m                              #Se centran los datos en el punto (0,0)
    dista = (coord_nuevo[:,0]**2+coord_nuevo[:,1]**2)**0.5     #Distancia del centro a los puntos
    maxi = math.ceil(max(dista))                   #Mayor de 'dista', transformado a entero
    vertices = np.array([[-maxi,maxi],[-maxi,-maxi],[maxi,-maxi],[maxi,maxi]])  #vértices del cuadrado
    coord_nuevo = coord_nuevo - vertices[0]              #Transladando los puntos al cuadrante 4
    vertices = vertices - vertices[0]              #Los vértices también se trasladan
    return coord_nuevo,vertices                    #Se retorna los valores


def m_imagen(coord,vertices,filename,pixeles=24):
    #Función para obtener las coordenadas en la matriz
    
    #Creación de la matriz
    size = (pixeles,pixeles)
    matriz = np.zeros(size)
    
    #Transformación de las coordenadas como indices para la matriz
    coord_m = (coord/vertices[2,0])*(pixeles-1)
    coord_m = np.round(abs(coord_m))
    
    #Rellenando con unos las posiciones de las caracteristicas
    for i,j in zip(coord_m[:,1],coord_m[:,0]):
        matriz[int(i),int(j)] = 1       
    
    #Condicional por si las caracteristicas fueron agrupadas en una misma posición
    if(np.count_nonzero(matriz!=0)!=coord.shape[0]):
        #print("Error, repetir el proceso, superposición de características")
        return coord_m, matriz, True
    else:
        #matplotlib.image.imsave(filename[:-4]+'_forma_general.png', matriz,cmap='binary') 
        return coord_m, matriz, False
    
def difuminacion(matriz, coordenada, distancia=0.1, pasos=3, amplificacion=np.pi, opcion='maximo'):
    #Función que realiza la difuminación
    
    x = int(coordenada[1])
    y = int(coordenada[0])
    valor_central = matriz[x,y]
    
    for p in range(pasos):
        r_actual = int(matriz.shape[0]*distancia*(p+1))   #radio actual  de mayor a menor
        
        #Funcion de intensidad
        intensidad=min(amplificacion*valor_central/(np.pi*r_actual**2),valor_central)
        
        #Delimitación del área
        lim_inf_i = max(x-r_actual-1,0)
        lim_sup_i = min(x+r_actual+1,matriz.shape[0])
        lim_inf_j = max(y-r_actual-1,0)
        lim_sup_j = min(y+r_actual+1,matriz.shape[1])
        
        #Colocación de valores
        for i in range(lim_inf_i, lim_sup_i):
            for j in range(lim_inf_j, lim_sup_j):
                if((x-i)**2 + (y-j)**2 <= r_actual**2):
                    if(matriz[i,j]==0):
                        matriz[i,j]=intensidad
                    elif(x!=i and y!=j): #Sobreposición
                        if(opcion=='mean'):
                            matriz[i,j]=(matriz[i,j]+intensidad)/2 
                        elif(opcion=='maximum'):
                            matriz[i,j]=max(matriz[i,j],intensidad)
    return matriz
    
    
def imagen_muestras(X, Y, coord, matriz, carpeta, amplificacion, distancia=0.1, pasos=3, opcion='maximo', train_m=False):
    #Función para crear imágenes
    
    #Aquí es donde se realiza el preprocesamiento siendo 'matriz' = 'm'
    if train_m:
        matriz_a = np.zeros(matriz.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
                matriz_a = difuminacion(matriz_a, coord[j], distancia, pasos, amplificacion, opcion)
        matriz = np.copy(matriz_a)
    
    #En esta parte se generan las imágenes por cada muestra
    for i in range(X.shape[0]):
        matriz_a = np.copy(matriz)
        
        for j in range(X.shape[1]):
            matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
            matriz_a = difuminacion(matriz_a, coord[j], distancia, pasos, amplificacion, opcion)
        
        for j in range(X.shape[1]):
            matriz_a[int(coord[j,1]),int(coord[j,0])]=X[i,j]
        
        extension = 'png'   #eps o pdf
        subcarpeta = str(int(Y[i])).zfill(2)    #subcarpeta para agrupar los resultados de cada clase
        nombre_imagen = str(i).zfill(6)
        ruta = os.path.join(carpeta,subcarpeta)
        ruta_completa = os.path.join(ruta,nombre_imagen+'.'+extension)
        if not os.path.isdir(ruta):            
            try:
                os.makedirs(ruta)
            except:
                print("Error: Could not create subfolder")
        matplotlib.image.imsave(ruta_completa, matriz_a, cmap='binary', format=extension)
        
        #Se utiliza solo en '.png', el resto es para obtener las imágenes en una mejor calidad.
        # matplotlib.image.imsave(carpeta+str(i).zfill(6)+'_'+str(int(Y[i])).zfill(2)+'.png', matriz_a,cmap='binary')
        #matplotlib.image.imsave(carpeta+str(i).zfill(6)+'_'+str(int(Y[i])).zfill(2)+'.eps', matriz_a,cmap='binary',format='eps')
        #matplotlib.image.imsave(carpeta+str(i).zfill(6)+'_'+str(int(Y[i])).zfill(2)+'.pdf', matriz_a,cmap='binary',format='pdf')
        #break
        
    #Se retorna la matriz 'm'.
    return matriz

def guardar_variable(X, filename='objs.pkl',verbose=False):
    #Función para guardar variable en un archivo
    with open(filename, 'wb') as f:
        pickle.dump(X, f)
    if(verbose):
        print("Se ha guardado correctamente en "+filename)

def cargar_variable(filename='objs.pkl',verbose=False):
    #Función para cargar variable de un archivo
    with open(filename, 'rb') as f:
        variable = pickle.load(f)
    if(verbose):
        print("Se ha cargado correctamente de "+filename)
    return variable

#####

#Clase para crear imágenes
class DataImg:
    
    def __init__(self, algoritmo='PCA', pixeles=20, random_seed=20, veces=4, amp=np.pi, distancia=0.1, pasos=4, opcion='maximo'):
        self.algoritmo = algoritmo      # algoritmo de reduccion dimensional
        self.p = pixeles                # pixeles de la imagen
        self.random_seed = random_seed  # semilla
        self.veces = veces              # solo para t-sne
        
        self.amp = amp              # amplitud (difuminacion)
        self.distancia = distancia  # distancia (difuminacion)
        self.pasos = pasos          # pasos (difuminacion)
        self.opcion = opcion        # opción de sobreposición (difuminacion)
        
        self.error_pos = False      # Indica la superposicion de pixeles caracteristicos
        
    def ObtenerCoord(self, X, verbose=False):
        # En esta función se utiliza los algoritmos de reducción dimensional
#+{@RETC
        self.min_max_scaler = MinMaxScaler()
##        self.X_minmax = self.min_max_scaler.fit_transform(X)
        X = self.min_max_scaler.fit_transform(X)
#+}@RETC
        labels = np.arange(X.shape[1])
        X_trans = X.T
        
        if(verbose):
            print("Algoritmo seleccionado: "+self.algoritmo)
            
        if(self.algoritmo=='PCA'):
            X_embedded = PCA(n_components=2,random_state=self.random_seed).fit(X_trans).transform(X_trans)
        elif(self.algoritmo=='t-SNE'):
            #print("inicio_iter")
            for i in range(self.veces):
                X_trans = np.append(X_trans,X_trans,axis=0)
                labels = np.append(labels,labels,axis=0)
            #print("inicio_tsne")
            X_embedded = TSNE(n_components=2,random_state=self.random_seed,perplexity=50).fit_transform(X_trans)
            #X_embedded = TSNE(n_components=2,perplexity=50).fit_transform(X_trans)
            #print("termino tsne")
        else:
            print("Error: Algoritmo incorrecto")
            X_embedded = np.random.rand(X.shape[1],2)
        
        datos_coordenadas = {'x':X_embedded[:,0], 'y':X_embedded[:,1], 'Sector':labels}
        dc = pd.DataFrame(data=datos_coordenadas)
        self.coord_obtenidas = dc.groupby('Sector').mean().values
#-{@RETC
#       self.min_max_scaler = MinMaxScaler()
#        self.X_minmax = self.min_max_scaler.fit_transform(X)
#-}@RETC
        del X_trans
        gc.collect()
    
    def Delimitacion(self):
        # En esta función se realiza la delimitación del área
        self.coordenadas_iniciales, self.vertices = cuadrado(self.coord_obtenidas)
        
    def ObtenerMatrizPosiciones(self, filename='original'):
        # En esta función se obtiene las posiciones en la matriz
        self.pos_pixel_caract, self.m, self.error_pos = m_imagen(self.coordenadas_iniciales,self.vertices,filename,pixeles=self.p)
        
    def CrearImg(self, X, Y, carpeta = 'prueba/', train_m=False, verbose=False):
        # En esta función se crean las imágenes que serán procesadas por la CNN
        
        # train_m es un booleano que indica si esta entrenamiento para realizar
        # el prepocesamiento para la matriz 'self.m'
        
        
        X_escalado = self.min_max_scaler.transform(X)
        Y = np.array(Y)
        try:
            os.mkdir(carpeta)
            if(verbose):
                print("Se creó la carpeta "+carpeta+"...")
        except:
            if(verbose):
                print("La carpeta "+carpeta+" ya esta creada...")
         
#+{@RETC para imprimir la posición de los píxeles en lás imágenes
        #print(self.pos_pixel_caract)
        #print(X_escalado)
#+}@RETC
        self.m = imagen_muestras(X_escalado, Y, self.pos_pixel_caract, self.m, carpeta, self.amp, distancia=self.distancia, pasos=self.pasos, opcion=self.opcion, train_m=train_m)
        
    def Entrenamiento(self, X, Y, carpeta = 'img_train/', verbose=False):
        #Esta función utiliza las anteriores funciones para el entrenamiento.
        #En este caso, se crean y modifican las variables.
        #print("Inicia_obtenercoord")
        self.ObtenerCoord(X, verbose=verbose)
        #print("Inicia delimitacion")
        self.Delimitacion()
        #print("inicia Obtener matriz")
        self.ObtenerMatrizPosiciones()
        #print("inicia crearImg")
        self.CrearImg(X, Y, carpeta, train_m=True, verbose=verbose)
        #print("termino crearImg")
        
    def Prueba(self, X, Y=None, carpeta = 'img_test/', verbose=False):
        #Esta función utiliza las anteriores funciones para la prueba o validación.
        #En este caso, solo se utilizan las variables anteriormente calculados en
        # 'Entrenamiento', no se crean y modifican las variables.
        
        if(Y is None):
            Y = np.zeros(X.shape[0])
        self.CrearImg(X, Y, carpeta, train_m=False, verbose=verbose) 


#Imprime los parametros
#config = vars(args)
#print(config)

##########
#Ejecución

#Blurring verification
if not args.blurr_active:
    args.amplification = 0
    args.distance = 0.1
    args.steps = 0
    # args.option = 'mean'


#Lectura de CSV
dataset = pd.read_csv(args.src_data)#, skiprows=[0])
array = dataset.values
if args.load_configuration:
    X = array
    modeloIMG = cargar_variable(filename=args.load_configuration,verbose=args.verbose)
    
    #Prueba
    modeloIMG.Prueba(X,carpeta=args.dest_folder,verbose=args.verbose)
else:
    X = array[:,:-1]
    Y = array[:,-1]
    #Creación objeto
    modeloIMG = DataImg(algoritmo=args.algorithm,
                        pixeles=args.pixels,
                        amp=args.amplification,
                        distancia=args.distance,
                        pasos=args.steps,
                        opcion=args.option,
                        random_seed=args.random_seed,
                        veces=args.times_tsne
                        )

    #Entrenamiento
    modeloIMG.Entrenamiento(X, Y, carpeta=args.dest_folder,verbose=args.verbose)

#Guada la configuracion para posterior uso
if args.save_configuration:
    guardar_variable(modeloIMG, filename=args.save_configuration,verbose=args.verbose)