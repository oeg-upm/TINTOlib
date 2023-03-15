## Data2Image

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/BorjaRei/Data2Image/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)
[![Documentation Status](https://readthedocs.org/projects/morph-kgc/badge/?version=latest)]()
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/)

**Data2Image** is a state-of-the-art library that wraps the most important techniques for the construction of **Synthetic Images** from [Sorted Data](https://www.jstatsoft.org/article/view/v059i10) (also known as **Tabular Data**). 

## Features
- Input data formats (2 options):
    - **Pandas Dataframe** 
    - **Files with the following format** 
        - **Tabular files**: The input data must be in **[CSV](https://en.wikipedia.org/wiki/Comma-separated_values)**, taking into account the **[Tidy Data](https://www.jstatsoft.org/article/view/v059i10)** format.
        - **Tidy Data**: The **target** (variable to be predicted) should be set as the last column of the dataset. Therefore, the first columns will be the features.
        - All data must be in numerical form.
        
- Runs on **Linux**, **Windows** and **macOS** systems.
- Compatible with **[Python](https://www.python.org/)** 3.7 or higher.

## Models

|  Model  |   Class   | Features |                                          Hyperparameters                                          |
|:-------:|:---------:|:--------:|:-------------------------------------------------------------------------------------------------:|
| [TINTO](https://github.com/oeg-upm/TINTO)| `tinto()` |  `blur`  | `algorithm` `pixels` `blur` `amplification` `distance` `steps` `option` `seed` `times` `verbose`  |

## Documentation

**[Read the documentation](https://data2image.readthedocs.io/en/latest/)**.

## Getting Started

**You can install Data2Image using [Pypi(test)](https://pypi.org/project/data2image-alpha/)**:

```
    pip install data2image-alpha
```


To import a specific model use 
```
    from data2Image.models import tinto
```

Create the model. If you don't set any hyperparameter, the model will use the default values ([read documentation](https://data2image.readthedocs.io/en/latest/)).
````
    model = tinto(blur=True)
````
To generate the synthetic images use ``.genereateImages(data,folder)`` method.
````
    model.generateImages(data, resultsFolderPath)
````

## License

Data2Image is available under the **[Apache License 2.0](https://github.com/BorjaRei/Data2Image/blob/main/LICENSE)**.

## Authors
- **[Borja Reinoso](https://github.com/borjarei) - [borjareinoso@gmail.com](borjareinoso@gmail.com)**
- **[Manuel Castillo-Cara](https://github.com/manwestc) - [jcastillo@fi.upm.es](mailto:jcastillo@fi.upm.es)**
- **[Raúl García-Castro](https://github.com/rgcmme)**



