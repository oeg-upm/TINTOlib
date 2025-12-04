# Standard library imports
import os
import pickle
from abc import ABC, abstractmethod
import warnings
from contextlib import nullcontext

# Third-party library imports
import pandas as pd
from tqdm import tqdm
# Typing imports
from typing import Optional, Union

# Default configuration values
default_problem = "classification"  # Define the type of task [classification, unsupervised, regression]
default_verbose = False         # Verbose: if True, shows the compilation text
default_hyperparameters_filename = 'objs.pkl'
allowed_values_for_problem = ["classification","supervised", "unsupervised", "regression"]

class AbstractImageMethod(ABC):
    """
    Abstract class that other classes must inherit from and implement abstract methods.
    Provides utility methods for saving/loading hyperparameters and data transformations.
    """
    def __init__(
        self,
        problem: Optional[str],
        verbose: Optional[bool],
        transformer=None
    ):
        # Validate `problem`
        if problem is None:
            problem = default_problem
        elif (not isinstance(problem, str)):
            raise TypeError(f"problem must be of type str (got {type(problem)})")
        elif(problem not in allowed_values_for_problem):
            raise ValueError(f"Allowed values for problem are {allowed_values_for_problem}. Instead got {problem}")

        if(problem == "supervised"):
            warnings.warn("Problem type supervised will be deprecated. Instead use classification.",FutureWarning)
        # Validate `verbose`
        if verbose is None:
            verbose = default_verbose
        if not isinstance(verbose, bool):
            raise TypeError(f"verbose must be of type bool (got {type(verbose)})")

        self.transformer=transformer
        self.problem = problem
        self.verbose = verbose
        self._fitted = False  # Tracks if fit has been called

    def saveHyperparameters(self, filename=default_hyperparameters_filename):
        """
        This function allows SAVING the transformation options to images in a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)
        if self.verbose:
            print(f"Hyperparameters successfully saved in {filename}.")

    def loadHyperparameters(self, filename=default_hyperparameters_filename):
        """
        This function allows LOADING the transformation options to images from a Pickle object.
        This point is basically to be able to reproduce the experiments or reuse the transformation
        on unlabelled data.
        """
        with open(filename, 'rb') as f:
            variables = pickle.load(f)
        
        for key, val in variables.items():
            setattr(self, key, val)

        if self.verbose:
            print(f"Hyperparameters successfully loaded from {filename}.")
        
    def fit(self, data):
        """
        Fits the model to the tabular data.

        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        """
        with (tqdm(total=100) if self.verbose == True else nullcontext()) as self.bar:
            self.__progress=0
            dataset = self._load_data(data)
            x, y = self._split_features_targets(dataset)
            # Transform features if required
            if self.transformer!=None:
                x = pd.DataFrame(self.transformer.fit_transform(x), columns=x.columns)

            self.__update_progress_bar(progress=20, text='Fitting process')
            # Call the training function
            self._fitAlg(x, y)
            self._fitted = True  # Mark as fitted
            self.__update_progress_bar(progress=80, text='Fit process completed')
            if self.verbose:
                self._write_message("Fit process completed.")

    def transform(self, data, folder):
        """
        Generate and saves the synthetic images in the specified folder.
        
        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        - folder: Path to folder where the images will be saved.
        """
        if not self._fitted:
            raise RuntimeError("The model must be fitted before calling 'transform'. Please call 'fit' first.")
        with (tqdm(total=100) if self.verbose==True else nullcontext()) as self.bar :
            self.__imgs_routes=[]
            self.__progress=0

            dataset = self._load_data(data)
            x, y = self._split_features_targets(dataset)
            self.bar_transform_step = (90 / x.shape[0])

            # Transform features if required
            if self.transformer != None:
                x = pd.DataFrame(self.transformer.transform(x), columns=x.columns)

            self.__update_progress_bar(progress=10, text='Generating and saving the synthetic images')
            self.folder = folder
            self._transformAlg(x, y)
            self.__create_csv(x,y)
            self.__update_progress_bar(progress=(100-self.__progress), text='Images generated and saved')
            self._write_message("Transform process completed.")

    def fit_transform(self, data, folder):
        """
        Fits the model to the tabular data and then generate and saves the synthetic images in the specified folder.
        
        Parameters:
        - data: Path to CSV file or a pandas DataFrame containing data and targets.
        - folder: Path to folder where the images will be saved.
        """


        with (tqdm(total=100,dynamic_ncols=True) if self.verbose==True else nullcontext()) as bar:
            self.__imgs_routes=[]
            self.bar=bar
            self.__progress = 0

            dataset = self._load_data(data)
            x, y = self._split_features_targets(dataset)
            self.bar_transform_step=(70/x.shape[0])

            # Transform features if required
            if self.transformer!=None:
                x = pd.DataFrame(self.transform.fit_transform(x), columns=x.columns)

            self.__update_progress_bar(progress=10, text='Generating and saving the synthetic images')
            self.folder = folder
            self._fitAlg(x, y)
            self.__update_progress_bar(progress=20, text='Generating and saving the synthetic images')
            self._transformAlg(x, y)
            self._fitted = True  # Mark as fitted after both operations
            self.__update_progress_bar(progress=(100-self.__progress), text='Images generated and saved')
            self._write_message("Fit-Transform process completed.")

    def _load_data(self, data: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Loads data from a file or returns the DataFrame directly.
        """
        self.__update_progress_bar(progress=0,text='Preparing Data')
        if isinstance(data, str):
            dataset = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            dataset = data
        else:
            raise TypeError("data must be a string (file path) or a pandas DataFrame.")

        if(dataset.select_dtypes(exclude=["number"]).shape[1]!=0):
            raise TypeError("There are non-numeric features in data")

        if self.verbose:
            self._write_message("Data successfully loaded.")

        return dataset

    def _split_features_targets(self, dataset: pd.DataFrame):
        """
        Splits dataset into features and targets based on the problem type.
        """
        if self.problem in ["classification","supervised", "regression"]:
            x = dataset.drop(columns=dataset.columns[-1])
            y = dataset[dataset.columns[-1]]
        else:
            x = dataset
            y = None
        return x, y

    def __get_image_routes(self, y, num_image):
        """
        Build image routes depending on problem type.
        Args:
            y: Class variable
            num_image: number of image that we'll build its routes

        Returns:
            (absolute route,relative route)
        """
        image_file = str(num_image).zfill(6) + ".png"
        if (self.problem in ["classification","supervised"]):
            subfolder = str(int(y)).zfill(2)
        else:
            subfolder = "images"

        folder_route = os.path.join(self.folder, subfolder)
        abs_route = os.path.join(self.folder, subfolder, image_file)
        rlt_route = os.path.join(subfolder, image_file)

        if not os.path.isdir(folder_route):
            try:
                os.makedirs(folder_route)
            except:
                self._write_message("Error: Could not create subfolder")

        return abs_route, rlt_route

    def _save_image(self, img_matrix, y,num_image):
        abs_route,rtv_route=self.__get_image_routes(y,num_image)
        self._img_to_file(img_matrix,abs_route,"png")
        self.__imgs_routes.append({"images":rtv_route,"value":y})
        self.__update_progress_bar(progress=self.bar_transform_step)

    def __create_csv(self,X,y):
        """
        Create csv file to save the images routes
        """
        filepath = self.folder + "/" + self.problem + ".csv"
        df = pd.concat([pd.DataFrame(data=self.__imgs_routes),X], axis=1)
        df.to_csv(filepath, index=False)

    def _features_pos_to_csv(self,columns,features_positions):
        filepath = self.folder + "/features_positions.csv"
        df = pd.DataFrame({"feature":columns,"row":features_positions[:,0],"column":features_positions[:,1]})
        df.to_csv(filepath, index=False)

    def __update_progress_bar(self,progress=None,text=None):
        if(self.verbose):
            if(progress != None):
                self.__progress = self.__progress + progress
                self.bar.update(progress)

            if(text != None):
                self.bar.set_description(text,refresh=True)

    def _write_message(self,message):
        if(self.verbose):
            self.bar.write(message)

    @abstractmethod
    def _fitAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Abstract method for fitting the algorithm. Must be implemented by subclasses.
        This method is not to be called from the outside.
        """
        raise NotImplementedError("Subclasses must implement _fit_alg.")

    @abstractmethod
    def _transformAlg(self, x: pd.DataFrame, y: Union[pd.DataFrame, None]):
        """
        Abstract method for transforming the data. Must be implemented by subclasses.
        This method is not to be called from the outside.
        """
        raise NotImplementedError("Subclasses must implement _transform_alg.")

    @abstractmethod
    def _img_to_file(self, image_matrix, file,extension):
        raise NotImplementedError("Subclasses must implement _img_to_file method.")
