import os
import numpy as np
import pandas as pd
from TINTOlib.mappingMethod import MappingMethod
import matplotlib
import shutil
import cv2
import TINTOlib.utils.constants as constants

###########################################################
################    MultiTransformer    ##############################
###########################################################

default_zoom=1
default_cmap='gray'
modes_allowed=[constants.stack_option,constants.avg_option]
preconditions_message=f"""The arguments for constructor must be a list of MappingMethod instances (2 or 3) using the same problem type, image dimension, output format and color map. 
                        The mode parameters must be in {modes_allowed}"""
invalid_input_shape="To combine data each case must have the same shape"
range_methods_availables_message="subfolders must be an array with 2 or 3 paths corresponding to images created with different transformers."
modes_allowed_message="mode must be either stack or avg."

class MultiTransformer():

    """
        MultiTransformer: A class for transforming tabular data into synthetic images using multiple transformers.

        Parameters:
        ----------
        imageTransformers : List
            The type of problem, defining how the images are grouped.
        mode : str
            Indicate if the result image is created averaging each original image channels or stacking the first channel of original images.
            Default is stack.
    """

    def __init__(self,imageTransformers,mode=constants.avg_option):
        super().__init__()
        self.__imageTransformers=imageTransformers
        self.__fitted=False
        self.__mode=mode

        if(isinstance(self.__imageTransformers,list)==False or self.__mode not in modes_allowed or len(self.__imageTransformers) not in [2, 3]):
            self.__preconditions_exception()

        problems = set()
        formats = set()
        maps = set()

        for transformer in self.__imageTransformers:
            if(not isinstance(transformer, MappingMethod)):
                self.__preconditions_exception()
            problems.add(transformer.problem)
            formats.add(transformer.format)
            maps.add(transformer.cmap)

        if (len(problems) != 1  or len(formats)!=1 or len(maps)!=1):
                self.__preconditions_exception()

        self.__problem=next(iter(problems))
        self.__input_format=next(iter(formats))
        self.__cmap=next(iter(maps))

    def __preconditions_exception(self):
        raise RuntimeError(preconditions_message)

    def fit(self, data):
        """
        fit each transformer
        Args:
            data: tabular dataset
        """
        for method in self.__imageTransformers:
            method.fit(data)
        self.__fitted = True

    def transform(self, data, folder, subfolders=None, delete_subfolders=False):
        """
        create images for each transformer and combine them.
        Args:
            data: tabular dataset
            folder: path to save images combined
            subfolders: path to save images by each transformer
            delete_subfolders: delete subfolders and images for each transformer after combine them.
        """
        if not self.__fitted:
            raise RuntimeError(constants.untrained_model_message)

        if isinstance(data, str):
            dataset = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            dataset = data
        else:
            raise TypeError(constants.invalid_input_data_message)

        X = dataset.drop(columns=dataset.columns[-1])
        y = dataset[dataset.columns[-1]]

        if (not os.path.exists(folder)):
            os.makedirs(folder)

        if (subfolders is None):
            subfolders = []
            for i in range(len(self.__imageTransformers)):
                subfolders.append(os.join.path(folder,"Method"+str(i)))

        for method,subfolder in zip(self.__imageTransformers,subfolders):
            method.transform(data,subfolder)

        num_images=X.shape[0]
        imgs_routes=[]

        for i in range(num_images):
            image = MultiTransformer.__read_image(y, i, subfolders,self.__problem,self.__input_format,self.__mode,len(self.__imageTransformers))
            img_route=MultiTransformer.__save_image(y,i,folder,image,self.__problem,self.__input_format)
            imgs_routes.append({constants.images_subfolder_name:img_route,constants.target_column_name:y[i]})

        MultiTransformer.__generate_csv(self.__problem,folder,imgs_routes,X)

        if(delete_subfolders):
            for subfolder in subfolders:
                shutil.rmtree(subfolder)

    @staticmethod
    def __read_image(y,i,folders,problem,input_format,mode,num_transformers):
        """
        Reads and combine images
        Args:
            y: target values
            i: image number
            folders: folders to read images to combine them
            problem: type of problem
            input_format: data format using to read as numpy(npy) or image (png)
            mode: Indicate if images are created stacking one channel or averaging each channel of transformers original images
            num_transformers: number of transformers to merge to create images
        """
        if problem in [constants.classification, constants.supervised]:
            img_name=os.path.join(str(int(y[i])).zfill(2),str(i).zfill(6)+"."+input_format)
        else:
            img_name=os.path.join(constants.images_subfolder_name,str(i).zfill(6)+"."+input_format)

        try:
            if(input_format==constants.png_format):
                if(mode==constants.stack_option):
                    if(num_transformers==3):
                        return np.dstack([np.array(cv2.imread(os.path.join(folder, img_name),cv2.IMREAD_GRAYSCALE)/255) for folder in folders])
                    else:
                        arr=[np.array(cv2.imread(os.path.join(folder, img_name),cv2.IMREAD_GRAYSCALE)/255) for folder in folders]
                        return np.dstack([arr[0],arr[1],np.mean(arr,axis=0)])
                else:
                    return np.mean([np.array(cv2.imread(os.path.join(folder, img_name),cv2.IMREAD_COLOR_RGB)/255) for folder in folders],axis=0)
            else:
                if(mode==constants.stack_option):
                    if(num_transformers==3):
                        return np.dstack([np.load(os.path.join(folder,img_name),allow_pickle=True) for folder in folders])
                    else:
                        arr = [np.load(os.path.join(folder,img_name),allow_pickle=True) for folder in folders]
                        return np.dstack([arr[0], arr[1], np.mean(arr, axis=0)])
                else:
                    return np.mean([np.load(os.path.join(folder,img_name),allow_pickle=True) for folder in folders],axis=0)
        except Exception as e:
            raise ValueError(invalid_input_shape) from None

    @staticmethod
    def __save_image(y,i,folder,image_matrix,problem,output_format):
        """
        Save combined images using the specified format
        Args:
            y: target values
            i: image number
            folder: folder to save image
            image_matrix: image data
            problem: type of problem
            output_format: format of output data
        """
        if problem in [constants.classification, constants.supervised]:
            img_folder = os.path.join(folder, str(int(y[i])).zfill(2))
            image_subfolder=str(int(y[i])).zfill(2)
        else:
            img_folder = os.path.join(folder, constants.images_subfolder_name)
            image_subfolder=constants.images_subfolder_name

        if (not os.path.exists(img_folder)):
            os.makedirs(img_folder)

        img_route=os.path.join(img_folder,str(i).zfill(6)+"."+output_format)

        if (output_format == constants.npy_format):
            np.save(img_route, image_matrix.astype(np.float64))
        else:
            matplotlib.image.imsave(img_route, image_matrix, format=output_format)

        return os.path.join(image_subfolder,str(i).zfill(6)+"."+output_format)

    @staticmethod
    def __generate_csv(problem,folder,imgs_routes,X):
        """
        Save CSV file with tabular data and image paths
        Args:
            problem: type of problem
            folder:path to save csv file
            imgs_routes:data structure with target values and images path
            X:features tabular data

        Returns:

        """
        filepath = folder + "/" + problem + ".csv"
        df = pd.concat([pd.DataFrame(data=imgs_routes),X], axis=1)
        df.to_csv(filepath, index=False)

    @staticmethod
    def merge_images(data,folder,subfolders,problem,mode,format):
        """
        Combine images generated with different transformers through folders paths.
        Args:
            data: tabular dataset
            folder: path to save images combined
            subfolders: path to save images by each transformer
            problem: type of problem
            mode: Indicate if images are created stacking one channel or averaging each channel of transformers original images
            format:  data format using to read/save as numpy(npy) or image (png)

        Returns:

        """

        if isinstance(data, str):
            dataset = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            dataset = data
        else:
            raise TypeError(constants.invalid_input_data_message)

        if(len(subfolders) not in [2,3]):
            raise ValueError(range_methods_availables_message)
        if(mode not in modes_allowed):
            raise ValueError(modes_allowed_message)

        X = dataset.drop(columns=dataset.columns[-1])
        y = dataset[dataset.columns[-1]]

        if (not os.path.exists(folder)):
            os.makedirs(folder)

        num_images = X.shape[0]
        imgs_routes = []
        for i in range(num_images):
            image = MultiTransformer.__read_image(y, i, subfolders, problem, format, mode,len(subfolders))
            img_route = MultiTransformer.__save_image(y, i, folder, image, problem, format)
            imgs_routes.append({constants.images_subfolder_name: img_route, constants.target_column_name: y[i]})

        MultiTransformer.__generate_csv(problem, folder, imgs_routes, X)