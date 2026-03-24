# Local application/library imports
from TINTOlib.abstractImageMethod import AbstractImageMethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from scipy.spatial.distance import cdist
from PIL import Image

import numpy as np
import pandas as pd

# Typing imports
from typing import Union

###########################################################
################    Clusters    ###########################
###########################################################

class Clusters(AbstractImageMethod):
    
    """
    Clusters: A class for transforming tabular data into synthetic images using unsupervised clustering techniques. 
    In a first phase, the data are transformed into clusters, and subsequently, images are generated based on these 
    clusters.
    
    Parameters:
    ----------
    problem : str, optional
        The type of problem, defining how the images are grouped. 
        
        Default is 'classification'. Valid values: ['classification', 'unsupervised', 'regression'].
    normalize : bool, optional
        If True, normalizes input data using MinMaxScaler. 
        
        Default is True. Valid values: [True, False].
    verbose : bool, optional
        Show execution details in the terminal. 
        
        Default is False. Valid values: [True, False].
    algorithm : str, optional
        Algorithm / technique to be applied for generating the synthetic image. 
        
        Default is 'kmeans'. Valid values: ["kmeans", "gaussianMix", "aggloKNN", "mixMethod", "kde", "kmedoids", "factor"].
    n_clusters : int / string / list of str, optional
        Number of clusters to be found to represent the data. The pixel matrix is defined as the square root of the number of clusters, rounded up to the nearest integer greater than that root, unless the root is exact.
        
        If this parameter is 'auto', it tries to obtain the most optimal number of clusters based on SSIM.
        
        If this parameter is a list of integers, the number of clusters used will be the most optimal one from that list based on SSIM.
        
        This parameter does not apply if the selected algorithm is KDE.
        
        Default is 16. Valid values: integer.
    random_seed : int, optional
        Seed for reproducibility.
        
        Default is 1. Valid values: integer.
    n_init : int / string, optional
        Number of initializations for initial selections. 
        
        This parameter only applies when using the “kmeans” and “gaussianMix” algorithms.
        
        Default is 'auto'. Valid values: integer or 'auto'.
    max_iter : int / string, optional
        Maximum number of iterations of the k-means, gaussian mixture and kmedoids algorithms for a single run.
        
        This parameter only applies when using the “kmeans”, “gaussianMix” and "kmedoids" algorithms.
        
        Default is 300. Valid values: integer.
    algorithmMethod : str, optional
        This parameter corresponds to the k-means algorithm parameter and can take the values ["lloyd", "elkan"].
        
        This parameter only applies when using the “k-means” algorithms.
        
        Default is 'lloyd'. Valid values: ["lloyd", "elkan"].
    covariance_type : str, optional
        This parameter corresponds to the gaussian mixture algorithm parameter and can take the values ['full', 'tied', 'diag', 'spherical'].
        
        This parameter only applies when using the “gaussianMix” algorithms.
        
        Default is 'full'. Valid values: ['full', 'tied', 'diag', 'spherical'].
    ensamMethod : list of str , optional
        List of allowed method names ["kmeans", "gaussianMix", "aggloKNN", "kmedoids", "factor"], where methods cannot be repeated. 
        
        The maximum size of the list is 3, and each method will correspond to a color channel of the image. If the list has fewer than 3 methods, the remaining higher channels will be filled with zero-filled matrices.
        
        Default is []. Valid values: ["kmeans", "gaussianMix", "aggloKNN", "kmedoids", "factor"]
    bandwidth : float , optional
        Bandwidth to be applied to each point when performing kernel density estimation.
        
        This parameter only applies when using the “kde” algorithms.
        
        Default is 1.0. Valid values: float.
    kernel : str , optional
        Type of kernel to be applied when performing kernel density estimation.
        
        This parameter only applies when using the “kde” algorithms.
        
        Default is 'gaussian'. Valid values: ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'].
    metric : str , optional
        Type of metric to compute distances for the "aggloKNN", "kmedoids" and "kde" algorithms.
        
        Default is 'euclidean'. Valid values: 
            - For 'kmedoids' and 'kde' : ['euclidean', 'manhattan', 'chebyshev'].
            - For 'aggloKNN' : ['euclidean', 'manhattan', 'cosine'].
            - For 'mixMethod' : ['euclidean', 'manhattan']
    RBFKmeans: boolean, optional
        If the k-means algorithm is used, we can choose whether the image is formed using the distances from the instances to each of the clusters, or by transforming these distances based on RBF. If this parameter is True, the conversion will be applied; if it is False, it will not be applied.
        
        Default False.
    """
    
    ALGORITHMS = {"kmeans", "gaussianMix", "aggloKNN", "mixMethod", "kde", "kmedoids", "factor"}
    ALGORITHMS_3CHANNELS = {"kmeans", "gaussianMix", "aggloKNN", "kmedoids", "factor"}
    ALGOTITHMS_OPTIMAL = {"kmeans", "gaussianMix", "mixMethod", "kmedoids", "factor"}
    default_ensamMethod = []
    COV_TYPES = {'full', 'tied', 'diag', 'spherical'}
    default_random_seed = 1
    default_n_clusters = 16
    default_algorithm = "kmeans"
    scale=StandardScaler()
    default_n_init = 'auto'
    default_max_iter = 300
    default_algorithmMethod = "lloyd"
    default_covariance_type = "full"
    default_bandwidth = 1.0
    default_kernel = "gaussian"
    KERNEL_TYPES={'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}
    METRICS_TYPES = {'euclidean', 'manhattan', 'chebyshev'}
    default_metric='euclidean'
    default_RBFKmeans = False
    
    def __init__(
        self,
        problem = None,
        normalize=None,
        verbose = None,
        algorithm = default_algorithm,
        n_clusters = default_n_clusters,
        random_seed=default_random_seed,
        n_init = default_n_init,
        max_iter = default_max_iter,
        algorithmMethod = default_algorithmMethod,
        covariance_type = default_covariance_type,
        ensamMethod = default_ensamMethod,
        bandwidth = default_bandwidth,
        kernel = default_kernel,
        metric=default_metric,
        RBFKmeans=default_RBFKmeans
        
    ):
        
        if algorithm=="aggloKNN":
            self.METRICS_TYPES = {'euclidean', 'manhattan', 'cosine'}
        
        if algorithm=="mixMethod":
            self.METRICS_TYPES = {'euclidean', 'manhattan'}
            
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{algorithm}'. "
                f"Allowed values are: {self.ALGORITHMS}"
            )
        
        if covariance_type not in self.COV_TYPES:
            raise ValueError(
                f"Invalid covariance_type '{covariance_type}'. "
                f"Allowed values are: {self.COV_TYPES}"
            )
        
        if kernel not in self.KERNEL_TYPES:
            raise ValueError(
                f"Invalid kernel '{kernel}'. "
                f"Allowed values are: {self.KERNEL_TYPES}"
            )
        
        if metric not in self.METRICS_TYPES:
            raise ValueError(
                f"Invalid metric '{metric}'. "
                f"Allowed values are: {self.METRICS_TYPES}"
            )
        
        if (algorithm=="mixMethod") and (len(ensamMethod)==0 or len(ensamMethod)>3):
            raise ValueError(
                "For this algorithm, the hyperparameter ensamMethod must have a length greater than zero and less than four."
            )
        
        if (algorithm=="mixMethod"):
            for k in range(len(ensamMethod)):
                if (ensamMethod[k] not in self.ALGORITHMS_3CHANNELS):
                    raise ValueError(
                        f"Algorithm type {ensamMethod[k]} not allowed for mixMethod; the allowed algorithms are: {self.ALGORITHMS_3CHANNELS}."
                    )
        
        if (algorithm=="mixMethod") and (len(ensamMethod) != len(set(ensamMethod))):
            raise ValueError(
                "Algorithms cannot be repeated."
            )
            
        if (algorithm=="kde" or algorithm=="aggloKNN") and (n_clusters=="auto" or isinstance(n_clusters, list)):
            raise ValueError(
                "For these algorithms ('kde', 'aggloKNN'), the 'auto' option to automatically select the optimal number of clusters is not enabled."
            )
        
        super().__init__(problem=problem, verbose=verbose)
        self.algorithm=algorithm
        self.n_clusters=n_clusters
        self.random_seed=random_seed
        self.n_init=n_init
        self.max_iter = max_iter
        self.algorithmMethod = algorithmMethod
        self.covariance_type = covariance_type
        self.ensamMethod = ensamMethod
        self.bandwidth=bandwidth
        self.kernel=kernel
        if metric=="manhattan":
            metric="cityblock"
        self.metric=metric
        self.RBFKmeans=RBFKmeans
    
    def _img_to_file(self, image_matrix, file):
        img = Image.fromarray(np.uint8(np.squeeze(image_matrix)))
        img.save(file)
    
    def __initialValues(self):
        
        self.modelList=[]
        self.minmaxList=[]
        self.ordenClusters=[]
        self.model=None
        self.minmax=None
        self.predictMix=[]
        self.kdeDistributions=[]
    
    def __xTransforImage(self,x):
        
        """
        - It handles two possible cases depending on the type of data stored in x.
        - x is list:
            - It will construct an image with three color channels.
            - Create an “empty” channel in case channels are missing.
            - For each block/channel k in the list, it normalizes it and converts it into a 2D image.
            - Scaling/normalization: The goal is to map values to an image-friendly range (typically 0–255).
            - Compute square dimension: We want to place the features on a square grid of size dim x dim.
            - Zero-padding to complete the square: It copies the real features at the beginning and leaves the rest as zeros.
            - Quantize and reshape into an image.
            - At the end, listArrayIn is a list of tensors with shape (n, dim, dim) (one per available channel in the list).
            - If there are fewer than 3 channels, pad them until reaching 3 channels.
            - Stack them into a single tensor with 3 channels.
        - x not is list:
            - Grayscale image (one channel).
            - Reorder the features if the algorithm is k-means or k-medoids. The goal is to keep “related” features close together when converting 
              them into an image.
            - Convert the feature vector into a 2D grid.
        """
        
        listArrayIn=[]
        
        if isinstance(x, list):
            emptyChanel=np.zeros((x[0].shape[0],int(np.ceil(np.sqrt(x[0].shape[1])))*int(np.ceil(np.sqrt(x[0].shape[1])))))
            for i,k in enumerate(x):
                xTrans=self.minmaxList[i].transform(k)
                dim=int(np.ceil(np.sqrt(xTrans.shape[1])))
                arrayIn=np.zeros((xTrans.shape[0],dim*dim))
                arrayIn[:,:xTrans.shape[1]] = xTrans
                xTrans=arrayIn.round().astype(np.uint8).reshape(-1,dim,dim)
                listArrayIn.append(xTrans)
            for i in range(len(x),3):
                listArrayIn.append(emptyChanel.round().astype(np.uint8).reshape(-1,dim,dim))
            x = np.stack([listArrayIn[0], listArrayIn[1], listArrayIn[2]], axis=-1)
        else:
            x=self.minmax.transform(x)        
            if (self.algorithm=="kmeans" or self.algorithm=="kmedoids"):
                order = np.array(self.ordenClusters, dtype=int)
                x = x[:, order]
            dim=int(np.ceil(np.sqrt(x.shape[1])))
            arrayIn=np.zeros((x.shape[0],dim*dim))
            arrayIn[:,:x.shape[1]] = x
            x=arrayIn.round().astype(np.uint8).reshape(-1,dim,dim)
        
        return x
    
    def __centroidsOrder(self,centroids):
        
        """
        - It computes the pairwise distances between every pair of centroids (Euclidean distance).
        - It builds a sequence by chaining together nearby centroids.
        - The goal is to fill self.ordenClusters with all cluster indices, ordered by proximity.
        - It starts the chain with the closest pair.
        - It filters the pairs to those that involve the current cluster.
        - It looks for the shortest pair among them.
        - Then it selects the “other end” of the pair, making sure it is not already in self.ordenClusters.
            - If the first index of the pair is not yet in the order, it adds it.
            - If it is already present, it adds the second one.
        - Then it removes pairs that involve the old index.
        - Overall, the goal is to build a path-like route that keeps jumping to the cluster closest to the last one added, while avoiding repeated clusters.
        """
        self.ordenClusters=[]
        listDist=[]
        for i in range(len(centroids)):
            for j in range(i+1,len(centroids)):
                distance=np.sqrt(np.sum((centroids[i]-centroids[j])**2))
                listDist.append([i,j,distance])
        listDist=np.array(listDist)
        
        ind=None
        while len(self.ordenClusters)<len(centroids):
            if ind is None:
                idx_min = np.argmin(listDist[:, 2])
                self.ordenClusters.append(listDist[idx_min,0])
                self.ordenClusters.append(listDist[idx_min,1])
                ind=listDist[idx_min,1]
                listDist=listDist[(listDist[:,0]!=listDist[idx_min,0]) & (listDist[:,1]!=listDist[idx_min,0])]
            else:
                listAux=listDist[(listDist[:,0]==ind) | (listDist[:,1]==ind)]
                idx_min = np.argmin(listAux[:, 2])
                if listAux[idx_min,0] not in self.ordenClusters:
                    self.ordenClusters.append(listAux[idx_min,0])
                    newInd=listAux[idx_min,0]
                else:
                    self.ordenClusters.append(listAux[idx_min,1])
                    newInd=listAux[idx_min,1]
                listDist=listDist[(listDist[:,0]!=ind) & (listDist[:,1]!=ind)]
                ind=newInd
    
    # pip install pyclustering
    def __kmedoids_pam(self,X, K, max_iter=100, random_state=0):
        
        """
        This function implements k-medoids using the PAM (Partitioning Around Medoids) algorithm in an iterative way: it selects K medoids 
        (real points from the dataset) and alternates between assigning each point to the nearest medoid and recomputing the best medoid within 
        each cluster, until convergence or the maximum number of iterations is reached.
            - Selects K distinct indices from 0..n−1 without replacement. Those points will be the initial medoids.
            - Each iteration has two phases: assignment and update.
            - Assignment phase:
                - cdist(X, X[medoid_indices]) computes a distance matrix.
                - np.argmin(D, axis=1) assigns each point to the nearest medoid.
                - It copies the current medoids; they will be replaced by better candidates as the algorithm proceeds.
            - Updata phase:
                - For each cluster k, it retrieves the indices of the points assigned to it.
                - If the cluster is empty (no points were assigned), it skips it.
                - intra_D: internal distance matrix of the cluster:
                    Shape (m, m) if the cluster has m points. intra_D[a, b] = distance between point a and point b within the cluster.
                - costs = intra_D.sum(axis=1):
                    For each candidate medoid (each point in the cluster), it computes its cost: cost(candidate j) = sum of distances from j to 
                    all other points in the cluster. This measures how “central” the point is (in the sense of minimizing total distance).
                - best = cluster_points[np.argmin(costs)]:
                    Selects as the new medoid the cluster point with the lowest total cost. By definition, this is the optimal medoid within the 
                    cluster for that distance metric.
                - Finally, it updates the medoid index for cluster k.
                - If, after recomputation, the medoids did not change, the algorithm has converged and stops. If they did change, it replaces them 
                  and proceeds to the next iteration.
            - Return values:
                - medoid_indices: final indices of the K medoids.
                - labels: final assignment of each point to its nearest cluster.
        """
        
        rng = np.random.RandomState(random_state)
        n = X.shape[0]
    
        # Inicialización
        medoid_indices = rng.choice(n, K, replace=False)        
        
        for _ in range(max_iter):
            # Asignación
            D = cdist(X, X[medoid_indices], metric=self.metric)
            labels = np.argmin(D, axis=1)
    
            new_medoids = medoid_indices.copy()
    
            # Actualización
            for k in range(K):
                cluster_points = np.where(labels == k)[0]
                if len(cluster_points) == 0:
                    continue
    
                intra_D = cdist(X[cluster_points], X[cluster_points], metric=self.metric)
                costs = intra_D.sum(axis=1)
                best = cluster_points[np.argmin(costs)]
                new_medoids[k] = best
    
            if np.all(new_medoids == medoid_indices):
                break
    
            medoid_indices = new_medoids
    
        return medoid_indices, labels
        
    def createImage(self, x , y, folder):
        
        """
        This function createImage(self, x, y, folder) takes tabular/numerical data (x) and converts it into “images” (2D matrices or 3D RGB-like tensors) 
        for each sample. It saves them to disk (via method _save_image) and finally generates a CSV file that maps image paths to their corresponding 
        label/value (when applicable).
            - Call xTranforImage. 
            - Save each image and record its file path.
        """
        Y = y.values if y is not None else None
        
        x=self.__xTransforImage(x)
        
        for i,sample in enumerate(x):
            self._save_image(sample,Y[i],i)
            
    def __kmeans(self,X,clustersIni, seedIni, typeProc=None):
        
        """
        1. It trains a scikit-learn KMeans model on X (storing it in self.model or self.modelList depending on typeProc), and it also prepares a MinMaxScaler to map distances to the centroids into the [0, 255] range (which is later used to turn those distances into “pixels” when creating images).
        2. If typeProc is None, it computes a cluster ordering (self.ordenClusters) based on how close the centroids are to each other, and then uses that ordering to reorder the “features” (centroid-distance features) so the image-like representation has more structure.
        
        - It creates the KMeans object using the parameters stored in self.
        - Training and creation of an “auxiliary dataset”: distances to centroids.
        - One channel case:
            - It trains KMeans on X.
            - It applies transform to X in order to generate the MinMaxScaler.
            - It scales those distances to an image-friendly range.
            - It obtains the centroids.
            - Calls the method that orders the centroids by proximity.            
        - Three channels case:
            - The main difference in how the data is handled compared to the single-channel case is that, when working with a list, we apply the trained model, the transform operation, and the MinMax scaling to each element of the list; additionally, with three channels, the column order is not modified.            
        """
        
        from sklearn.cluster import KMeans
                
        kmeans = KMeans(n_clusters=clustersIni, random_state=seedIni, n_init=self.n_init, max_iter=self.max_iter, algorithm=self.algorithmMethod) 
        
        if typeProc is None:
            self.model=kmeans.fit(X)
            auxX = self.model.transform(X)
            if self.RBFKmeans:
                sigma = np.mean(auxX)
                auxX = np.exp(-(auxX**2)/(2*sigma**2))            
            self.minmax=MinMaxScaler(feature_range=(0,255))
            self.minmax=self.minmax.fit(auxX)
        else:
            self.modelList.append(kmeans.fit(X))
            auxX = self.modelList[len(self.modelList)-1].transform(X)
            if self.RBFKmeans:
                sigma = np.mean(auxX)
                auxX = np.exp(-(auxX**2)/(2*sigma**2))                
            self.minmaxList.append(MinMaxScaler(feature_range=(0,255)))
            self.minmaxList[len(self.minmaxList)-1]=self.minmaxList[len(self.minmaxList)-1].fit(auxX)        
        
        if typeProc is None:
            centroids = self.model.cluster_centers_            
            self.__centroidsOrder(centroids)
            
            
    def __gaussianMix(self,X,clustersIni, seedIni, typeProc=None):
        
        """
        It trains a scikit-learn Gaussian Mixture Model (GMM) and prepares a MinMaxScaler to map into [0, 255] using some “auxiliary features” that, in this case, are not distances but membership probabilities for each component.
        
        For the grayscale case it uses a single, simple model type; when using more than one channel and mixing multiple algorithms, it uses a list-type structure where each position stores one model per algorithm.
        """

        from sklearn.mixture import GaussianMixture
        
        n_init = 1 if self.n_init == "auto" else self.n_init
        gaussianMixVar = GaussianMixture(n_components=clustersIni, n_init=n_init, random_state=seedIni, max_iter=self.max_iter, covariance_type=self.covariance_type) 
        
        if typeProc is None:
            self.model=gaussianMixVar.fit(X)
            auxX = self.model.predict_proba(X)
            self.minmax=MinMaxScaler(feature_range=(0,255))
            self.minmax=self.minmax.fit(auxX)
        else:
            self.modelList.append(gaussianMixVar.fit(X))
            auxX = self.modelList[len(self.modelList)-1].predict_proba(X)
            self.minmaxList.append(MinMaxScaler(feature_range=(0,255)))
            self.minmaxList[len(self.minmaxList)-1]=self.minmaxList[len(self.minmaxList)-1].fit(auxX)
    
    def __factor(self,X,clustersIni, seedIni, typeProc=None):
        
        """
        It trains a scikit-learn Factor Analysis and prepares a MinMaxScaler to map into [0, 255] using some “auxiliary features” that, in this case, are not distances but membership probabilities for each component.
        
        For the grayscale case it uses a single, simple model type; when using more than one channel and mixing multiple algorithms, it uses a list-type structure where each position stores one model per algorithm.
        """

        from sklearn.decomposition import FactorAnalysis
        
        factorVar = FactorAnalysis(n_components=clustersIni, random_state=seedIni) 
        
        if typeProc is None:
            self.model=factorVar.fit(X)
            auxX = self.model.transform(X)
            self.minmax=MinMaxScaler(feature_range=(0,255))
            self.minmax=self.minmax.fit(auxX)
        else:
            self.modelList.append(factorVar.fit(X))
            auxX = self.modelList[len(self.modelList)-1].transform(X)
            self.minmaxList.append(MinMaxScaler(feature_range=(0,255)))
            self.minmaxList[len(self.minmaxList)-1]=self.minmaxList[len(self.minmaxList)-1].fit(auxX)
        
    def __aggloKNN(self,X,clustersIni,typeProc=None):
        
        """
        Is a two-phase hybrid approach:

        Unsupervised clustering using Agglomerative Clustering (hierarchical), with a k-NN connectivity constraint to restrict which points can be merged.

        Using the labels obtained from the clustering, it trains a supervised KNN model in order to compute predict_proba; those probabilities are then used as auxiliary features (later scaled to [0, 255]).
        
        This algorithm can be part of the mix of algorithms used for three color channels. Therefore, in the grayscale case the model is stored as a single (simple) object, whereas when it is represented with three color channels, it is stored in a list.
        """
        
        from sklearn.neighbors import kneighbors_graph
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import KNeighborsClassifier
        
        Xnorm = normalize(X, norm="l2")
        
        n_conn = min(10, Xnorm.shape[0] - 1)
        n_conn = max(n_conn, 1)
        
        connectivity = kneighbors_graph(Xnorm, n_neighbors=n_conn, metric=self.metric)
        
        agg = AgglomerativeClustering(n_clusters=clustersIni,linkage="complete",connectivity=connectivity, metric=self.metric)
        
        labels_train = agg.fit_predict(Xnorm)
        
        n_knn = min(clustersIni * 3, Xnorm.shape[0] - 1)
        n_knn = max(n_knn, 1)
        
        knn = KNeighborsClassifier(
            n_neighbors=n_knn,        
            weights="uniform", 
            metric=self.metric,
            algorithm="brute"
        )
        
        if typeProc is None:
            self.model=knn.fit(Xnorm, labels_train)
            auxX = self.model.predict_proba(Xnorm)
            self.minmax=MinMaxScaler(feature_range=(0,255))
            self.minmax=self.minmax.fit(auxX)
        else:
            self.modelList.append(knn.fit(Xnorm, labels_train))
            auxX = self.modelList[len(self.modelList)-1].predict_proba(Xnorm)
            self.minmaxList.append(MinMaxScaler(feature_range=(0,255)))
            self.minmaxList[len(self.minmaxList)-1]=self.minmaxList[len(self.minmaxList)-1].fit(auxX)
    
    def __mixMethod(self,X,clustersIni, seedIni):
        
        """
        For each position in the ensamMethod hyperparameter list, a model is generated according to the algorithm chain specified at that position.
        """
        if "factor" in self.ensamMethod and clustersIni > len(X[0])-1:
            raise ValueError(
                "The number of clusters cannot be greater than the number of features for the factor algorithm."
            )
            
        for methodIn in self.ensamMethod:
            if (methodIn=="aggloKNN"):
               self.__aggloKNN(X,clustersIni,"mix")
            elif (methodIn=="kmeans"):
               self.__kmeans(X,clustersIni,seedIni,"mix") 
            elif (methodIn=="gaussianMix"):
               self.__gaussianMix(X,clustersIni,seedIni,"mix")
            elif (methodIn=="kmedoids"):
               self.__kmedoids(X,clustersIni,seedIni,"mix")
            elif (methodIn=="factor"):
               self.__factor(X,clustersIni,seedIni,"mix")
    
    def __kde(self,X):
        """
        This kde method builds a density-based representation of the data using Kernel Density Estimation (KDE) for each feature (column), and prepares that representation for image conversion (by scaling it to [0, 255]).

        For each feature in X:
            - It learns a one-dimensional density distribution (KDE).
            - It evaluates the density of each observed value under that distribution.
            - It uses those density values as new features.
            - It scales the result to [0, 255] so it can be treated as pixel intensity.

        The final result is a matrix where each value indicates how probable a given feature value is according to its estimated distribution.
        
        This algorithm cannot be used as a component of the three-color-channel representation because the image dimension depends on the data itself and cannot be controlled via clusters.
        """
        from sklearn.neighbors import KernelDensity
            
        kdeData = pd.DataFrame([])
        dfX=pd.DataFrame(X)
        for i,col in enumerate(dfX.columns):
            
            data = dfX[col].dropna().to_numpy().reshape(-1, 1)
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel, metric=self.metric)
            kde.fit(data)
            self.kdeDistributions.append(kde)
        
        kde_dict = {}
        
        from joblib import Parallel, delayed
        
        kdes = self.kdeDistributions
        cols = list(dfX.columns)
        
        def compute_col(i, col):
            
            data = dfX[col].to_numpy(dtype=float, copy=False)
            data = data[~np.isnan(data)]
            if data.size == 0:
                return str(i), np.array([])
            scores = np.exp(kdes[i].score_samples(data[:, None]))
            return str(i), scores
        
        results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(compute_col)(i, col) for i, col in enumerate(cols)
        )
        
        kde_dict = dict(results)
        
        
        kdeData = pd.DataFrame(kde_dict)
        
        self.minmax=MinMaxScaler(feature_range=(0,255))
        self.minmax=self.minmax.fit(np.array(kdeData))
        
    def __kmedoids(self,X, clustersIni, seedIni,typeProc=None):
        
        """
        Its main function is to:

        Run k-medoids (PAM) to obtain the medoids and the assignments.

        Build a per-sample representation based on distances to the medoids.

        Prepare a MinMaxScaler [0, 255] to convert those distances into image-like intensities.

        Compute a cluster ordering to improve the spatial structure of the image when using grayscale.
        """
        
        medoid_indices, labels = self.__kmedoids_pam(
            X,
            K=clustersIni,
            max_iter=self.max_iter,
            random_state=seedIni
        )
        
        if typeProc is None:
            self.model=X[medoid_indices]
            auxX = cdist(X, self.model, metric=self.metric)
            self.minmax=MinMaxScaler(feature_range=(0,255))
            self.minmax=self.minmax.fit(auxX)
        else:
            self.modelList.append(X[medoid_indices])
            auxX = cdist(X, self.modelList[len(self.modelList)-1], metric=self.metric)
            self.minmaxList.append(MinMaxScaler(feature_range=(0,255)))
            self.minmaxList[len(self.minmaxList)-1]=self.minmaxList[len(self.minmaxList)-1].fit(auxX)
        
        if typeProc is None:
            centroids = self.model            
            self.__centroidsOrder(centroids)
            
    # pip install scikit-image
    def __stability_score(self, images, sizeRes):
        
        """
        Calculate an average stability measure across a set of images using the Structural Similarity Index (SSIM).
        
        SSIM returns a value between:            
            - 1.0 identical images.
            - 0.0 no structural similarity.
            - < 0 very different images.
        - Window size must be at least 3.
        - SSIM requires the window size to be odd.
        - All unique combinations of image pairs are compared.
        """
        
        from skimage.metrics import structural_similarity as ssim
        
        if sizeRes < 3:
            raise ValueError("k must be >= 3")
        win_size=sizeRes if sizeRes % 2 == 1 else sizeRes - 1
        
        scores = []
        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                img1, img2 = images[i], images[j]
        
                # Si es RGB: shape (H, W, 3)
                if img1.ndim == 3:
                    s = ssim(img1, img2, data_range=255, channel_axis=-1, win_size=int(win_size))
                else: # Grises: shape (H, W)
                    s = ssim(img1, img2, data_range=255, win_size=int(win_size))
                
                scores.append(s)
        
        return float(np.mean(scores)) if scores else 1.0
    
    def __optimalNClusters(self, X, method, Ks=None, n_seeds=4, typeProc=None):
        
        """
        It attempts to choose the “optimal” number of clusters K (from a list of candidate Ks) using a stability criterion: if changing the random seed results in a clustering outcome that remains similar, then that K is considered more stable and therefore better.
        
        For each candidate K, it runs the algorithm multiple times with different seeds 
            - Converts each run’s result into “images”.
            - Measures how similar those images are across seeds (SSIM).
            - Averages the scores.
            - Selects the K with the highest stability.
        """        
        
        if Ks is None:
            Ks = [9, 16, 25, 36, 49, 64, 81]        
        
        if method=="factor":
            Ks = [val for val in Ks if val < len(X[0])-1]
        if method=="mixMethod" and "factor" in self.ensamMethod:
            Ks = [val for val in Ks if val < len(X[0])-1]            
        
        seeds = list(range(n_seeds))
        stability_by_K = []
        xOpt=X[:10000]
        if (method=="mixMethod"):
            xOpt=X[:2000]
        for K in Ks:
            
            # 1) Generar el conjunto completo de imágenes para cada seed
            imgs_by_seed = []
            for seed in seeds:
                
                self.__initialValues()
                
                if method=="kmeans":
                    self.__kmeans(xOpt, K, seed)
                    XEX = self.model.transform(xOpt)
                    if self.RBFKmeans:
                        sigma = np.mean(XEX)
                        XEX = np.exp(-(XEX**2)/(2*sigma**2))
                elif method=="gaussianMix":
                    self.__gaussianMix(xOpt, K, seed)
                    XEX = self.model.predict_proba(xOpt)
                elif method=="factor":
                    self.__factor(xOpt, K, seed)
                    XEX = self.model.transform(xOpt)
                elif method=="kmedoids":
                    self.__kmedoids(xOpt, K, seed)
                    XEX = cdist(xOpt, self.model, metric=self.metric)
                elif method=="mixMethod":
                    self.__mixMethod(xOpt, K, seed)
                    self.predictMix=[]
                    for i,k in enumerate(self.ensamMethod):
                        if k=="kmeans":
                            XEX=self.modelList[i].transform(xOpt)
                            if self.RBFKmeans:
                                sigma = np.mean(XEX)
                                XEX = np.exp(-(XEX**2)/(2*sigma**2))
                            self.predictMix.append(XEX)                            
                        elif k=="gaussianMix":
                            self.predictMix.append(self.modelList[i].predict_proba(xOpt))
                        elif k=="factor":
                            self.predictMix.append(self.modelList[i].transform(xOpt))
                        elif k=="aggloKNN":
                            xNorm = normalize(xOpt, norm="l2")
                            self.predictMix.append(self.modelList[i].predict_proba(xNorm))
                        elif k=="kmedoids":
                            self.predictMix.append(cdist(xOpt, self.modelList[i], metric=self.metric))
                    XEX = self.predictMix
                
                XEX=self.__xTransforImage(XEX)
                
                imgs = XEX
                # imgs debe ser array: (N,H,W) o (N,H,W,3)
                imgs_by_seed.append(imgs)
                
            # 2) Estabilidad por instancia
            N = imgs_by_seed[0].shape[0]
            per_instance_scores = []
            
            for i in range(N):
                images_i = [imgs_by_seed[s][i] for s in range(len(seeds))]
                per_instance_scores.append(self.__stability_score(images_i,int(np.ceil(np.sqrt(K)))))
        
            stability_by_K.append(float(np.mean(per_instance_scores)))
        
        # K óptimo = el de mayor estabilidad
        best_idx = int(np.argmax(stability_by_K))
        best_K = Ks[best_idx]
        
        return best_K, dict(zip(Ks, stability_by_K))
        
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        
        X = x.values
        x=self.scale.fit_transform(X)
        
        if self.n_clusters=="auto" or isinstance(self.n_clusters,list):
            
            self.__initialValues()
            
            Ks=None
            if isinstance(self.n_clusters,list):
                if (all(type(val) is int for val in self.n_clusters)):
                    Ks=self.n_clusters
                else:
                    raise ValueError(
                        "The list of cluster numbers to be tested as optimal must consist of integer values."
                    )        
            
            if (self.algorithm in self.ALGOTITHMS_OPTIMAL):
                self.n_clusters,*_ = self.__optimalNClusters(x, self.algorithm, Ks, n_seeds=6)            
            else:
                print(f"An admission clustering algorithm must be selected: {self.ALGOTITHMS_OPTIMAL}")
            
        self.__initialValues()
        
        if (self.algorithm=="kmeans"):
            self.__kmeans(x,self.n_clusters,self.random_seed)
        elif (self.algorithm=="gaussianMix"):
            self.__gaussianMix(x,self.n_clusters,self.random_seed)
        elif (self.algorithm=="aggloKNN"):
            self.__aggloKNN(x,self.n_clusters)
        elif (self.algorithm=="kde"):
            self.__kde(x)
        elif (self.algorithm=="kmedoids"):
            self.__kmedoids(x,self.n_clusters,self.random_seed)
        elif (self.algorithm=="mixMethod"):
            self.__mixMethod(x,self.n_clusters,self.random_seed)
        elif (self.algorithm=="factor"):
            if self.n_clusters > len(x[0])-1:
                raise ValueError(
                    "The number of clusters cannot be greater than the number of features - 1."
                )
            self.__factor(x,self.n_clusters,self.random_seed)
        else:
            print(f"An admission clustering algorithm must be selected: {self.ALGORITHMS}")

    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        
        x = self.scale.transform(x.to_numpy())
        if (self.algorithm=="kmeans"):
            x = self.model.transform(x)
            if self.RBFKmeans:
                sigma = np.mean(x)
                x = np.exp(-(x**2)/(2*sigma**2))
        elif (self.algorithm=="gaussianMix"):
            x = self.model.predict_proba(x)
        elif (self.algorithm=="factor"):
            x = self.model.transform(x)
        elif (self.algorithm=="aggloKNN"):
            xNorm = normalize(x, norm="l2")
            x = self.model.predict_proba(xNorm)
        elif (self.algorithm=="kde"):
            kdeData = []
            dfX=pd.DataFrame(x)
            kde_dict = {}
            
            from joblib import Parallel, delayed
            kdes = self.kdeDistributions
            cols = list(dfX.columns)
            
            def compute_col(i, col):
                
                data = dfX[col].to_numpy(dtype=float, copy=False)
                data = data[~np.isnan(data)]
                if data.size == 0:
                    return str(i), np.array([])
                scores = np.exp(kdes[i].score_samples(data[:, None]))
                return str(i), scores
            
            results = Parallel(n_jobs=-1, prefer="processes")(
                delayed(compute_col)(i, col) for i, col in enumerate(cols)
            )
            
            kde_dict = dict(results)
            
            kdeData = pd.DataFrame(kde_dict)
            
            x=kdeData.to_numpy()
        elif (self.algorithm=="kmedoids"):
            x = cdist(x, self.model, metric=self.metric)
        elif (self.algorithm=="mixMethod"):
            self.predictMix=[]
            for i,k in enumerate(self.ensamMethod):
                if k=="kmeans":
                    xIn=self.modelList[i].transform(x)
                    if self.RBFKmeans:
                        sigma = np.mean(xIn)
                        xIn = np.exp(-(xIn**2)/(2*sigma**2))
                    self.predictMix.append(xIn)
                elif k=="gaussianMix":
                    self.predictMix.append(self.modelList[i].predict_proba(x))
                elif k=="factor":
                    self.predictMix.append(self.modelList[i].transform(x))
                elif k=="aggloKNN":
                    xNorm = normalize(x, norm="l2")
                    self.predictMix.append(self.modelList[i].predict_proba(xNorm))
                elif k=="kmedoids":
                    self.predictMix.append(cdist(x, self.modelList[i], metric=self.metric))
            x = self.predictMix
        if (y is None):
            y = np.zeros(x.shape[0])
        self.createImage(x, y, self.folder)
