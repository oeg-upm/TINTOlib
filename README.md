# TINTOlib: Python Library to convert Tabular Data into Synthetic Images

[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/oeg-upm/TINTOlib-Documentation/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://pypi.python.org/pypi/)
[![Documentation Status](https://readthedocs.org/projects/morph-kgc/badge/?version=latest)](https://tintolib.readthedocs.io/en/latest/)
[![Open In Colab-CNN](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oeg-upm/TINTOlib-Crash_Course/blob/main/Notebooks/Challenge/Regression_CNN.ipynb)
[![Open In Colab-CNN+MLP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oeg-upm/TINTOlib-Crash_Course/blob/main/Notebooks/Challenge/Regression_CNN%2BMLP.ipynb)
[![Open In Colab-ViT](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oeg-upm/TINTOlib-Crash_Course/blob/main/Notebooks/Challenge/Regression_ViT.ipynb)
[![Open In Colab-ViT+MLP](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oeg-upm/TINTOlib-Crash_Course/blob/main/Notebooks/Challenge/Regression_ViT%2BMLP.ipynb)

<div>
    <p align = "center">
    <img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo.svg" alt="TINTO Logo" width="150">
    </p>
</div>

## 🎉 New Free Course on Udemy! (Spanish) 🎉

**We’ve just launched a 100% free course on Udemy** about **using TINTOlib** and developing **Hybrid Neural Networks**.

Learn how to turn tabular data into synthetic images and apply CNNs, ViTs, and hybrid architectures like a pro.

<p align="center">
  <a href="https://www.udemy.com/course/tintolib-deep-learning-tabutar-data-con-imagenes-sinteticas/?referralCode=16B7C59C2E3B0BD249D0" target="_blank">
    <img src="https://img.shields.io/badge/Udemy-Free%20Course-blueviolet?style=for-the-badge&logo=Udemy&logoColor=white" alt="Access the Course on Udemy"/>
  </a>
</p>

---

### 📺 VideoTutorial Course (English/Spanish)

🎥 Prefer not to register on Udemy or looking for the English version of the course? No worries — you can follow the full course directly on GitHub!

This hands-on tutorial includes **bilingual videos (English/Spanish)** and **practical notebooks** to help you learn how to use **TINTOlib** with deep learning models like CNNs, ViTs, and hybrid architectures.

<p align="center">
  <a href="https://github.com/oeg-upm/TINTOlib-Crash_Course" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-VideoTutorial%20Course-black?style=for-the-badge&logo=GitHub&logoColor=white" alt="Access the Course on GitHub"/>
  </a>
</p>


---

## 🧠 Overview

**TINTOlib** is a state-of-the-art Python library that transforms **tidy data** (also known as tabular data) into **synthetic images**, enabling the application of advanced deep learning techniques, including **Vision Transformers (ViTs)** and **Convolutional Neural Networks (CNNs)**, to traditionally structured data. This transformation bridges the gap between tabular data and powerful vision-based machine learning models, unlocking new possibilities for tackling regression, classification, and other complex tasks.

<!--
**Citing TINTO**: If you used TINTO in your work, please cite the **[SoftwareX](https://doi.org/10.1016/j.softx.2023.101391)**:

```bib
@article{softwarex_TINTO,
    title = {TINTO: Converting Tidy Data into Image for Classification with 2-Dimensional Convolutional Neural Networks},
    journal = {SoftwareX},
    author = {Manuel Castillo-Cara and Reewos Talla-Chumpitaz and Raúl García-Castro and Luis Orozco-Barbosa},
    volume={22},
    pages={101391},
    year = {2023},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2023.101391}
}
```

And use-case developed in **[INFFUS Paper](https://doi.org/10.1016/j.inffus.2022.10.011)** 

```bib
@article{inffus_TINTO,
    title = {A novel deep learning approach using blurring image techniques for Bluetooth-based indoor localisation},
    journal = {Information Fusion},
    author = {Reewos Talla-Chumpitaz and Manuel Castillo-Cara and Luis Orozco-Barbosa and Raúl García-Castro},
    volume = {91},
    pages = {173-186},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2022.10.011}
}
```
-->
---

## 📚 Features
- Input formats: **CSV** or Pandas DataFrame
- Designed for tidy data (**target column last**)
- Output: grayscale images from reduction and transformation methods
- Compatible with **Linux, Windows, macOS**
- Requires **Python 3.7+**

---


## 🧪 Methods
TINTOlib includes a variety of methods for generating synthetic images. Below is a summary of the supported methods and their hyperparameters:

| Methods | Class | Hyperparameters |
|:----------------------------------------------------------------:|:------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [TINTO](https://github.com/oeg-upm/TINTO) | `TINTO()` | `problem` `normalize` `verbose` `pixels` `algorithm` `blur` `submatrix` `amplification` `distance` `steps` `option` `times` `train_m` `zoom` `random_seed` |
| [IGTD](https://github.com/zhuyitan/igtd) | `IGTD()` | `problem` `normalize` `verbose` `scale` `fea_dist_method` `image_dist_method` `error` `max_step` `val_step` `switch_t` `min_gain` `zoom` `random_seed` |
| [REFINED](https://github.com/omidbazgirTTU/REFINED) | `REFINED()` | `problem` `normalize` `verbose` `hcIterations` `n_processors` `zoom` `random_seed` |
| [BarGraph](https://github.com/anuraganands/Non-image-data-classification-with-CNN/) | `BarGraph()` | `problem` `normalize` `verbose` `pixel_width` `gap` `zoom` |
| [DistanceMatrix](https://github.com/anuraganands/Non-image-data-classification-with-CNN/) | `DistanceMatrix()` | `problem` `normalize` `verbose` `zoom` |
| [Combination](https://github.com/anuraganands/Non-image-data-classification-with-CNN/) | `Combination()` | `problem` `normalize` `verbose` `zoom` |
| [SuperTML](https://github.com/GilesStrong/SuperTML_HiggsML_Test) | `SuperTML()` | `problem` `normalize` `verbose` `pixels` `feature_importance` `font_size` `random_seed` |
| [FeatureWrap](https://link.springer.com/chapter/10.1007/978-3-319-70139-4_87) | `FeatureWrap()` | `problem` `normalize` `verbose` `size` `bins` `zoom` |
| [BIE](https://ieeexplore.ieee.org/document/10278393) | `BIE()` | `problem` `normalize` `verbose` `precision` `zoom` |

---

## ⚠️ Platform-Specific Requirements for Certain Transformation Methods

Some transformation methods in TINTOlib have specific system requirements or limitations when used on platforms such as Google Colab, Windows, Linux, or macOS.

### REFINED

This method relies on `mpi4py`, which enables parallel computation using MPI (Message Passing Interface). However, `mpi4py` requires administrative permissions to utilize multiple processors, making it incompatible with platforms like Google Colab. 

- **Linux**:
  Ensure that the MPI environment is set up before installing `mpi4py`. Run the following commands:

    ```bash
      sudo apt-get install python3
      sudo apt install python3-pip
      sudo apt install python3-mpi4py
    ```
  Once MPI is installed:

    ```bash
      pip install mpi4py
    ```
- **MacOS / Windows:** Direct installation is usually supported:

    ```bash
      pip install mpi4py
    ```

### SuperTML

The **SuperTML** method generates text-based synthetic images and requires the **MS Sans Serif** font.

- On **Windows**, this font is typically available by default.
- On **Linux** and **macOS**, it must be installed manually to avoid rendering issues.

#### Font Installation

- **Linux**: Install Microsoft Core Fonts:
  
    ```bash
      sudo apt install ttf-mscorefonts-installer
    ```

On **Google Colab**, installing additional fonts is not permitted due to administrative restrictions.

## 📄 Getting Started

- You can install TINTOlib using **[Pypi](https://pypi.org/project/TINTOlib/)**:
    ```bash
      pip install TINTOlib
    ```

TINTOlib already includes all necessary dependencies, so there’s no need to install them individually.

However, if you prefer manual installation or want to explore the full environment:

- The repository includes a `requirements.txt` file listing the **core dependencies** required to use TINTOlib. You can directly run the **TINTOlib-example.ipynb** notebook located in the examples/ folder using the dependencies listed in `requirements.txt`.
- **Other notebooks**, which include training deep learning models on the generated images, require additional libraries. To run them, install the extended dependencies from `requirements-example.txt`:

---

## 🧩 Importing a Specific Model

- To use a specific image transformation model, import it directly. For example, to use **TINTO**:

    ```python
      from TINTOlib.tinto import TINTO
    ```

- To import a specific model use: 
    ```python
      from TINTOlib.tinto import TINTO
    ```

- Create the model. If you don't set any hyperparameter, the model will use the default values, refer to the **[Models Section](#models)** or the **[TINTO Documentation](https://tintolib.readthedocs.io/en/latest/)**.

    ```python
      model = TINTO(problem="supervised", blur=True)
    ```

---

## 🔧 Generating Synthetic Images
To generate synthetic images, use the following workflow with the `fit`, `transform`, and `fit_transform` methods:

**Parameters**:
- **data**: A path to a CSV file or a Pandas DataFrame  (target column must be the last column).
- **folder**: Path to the folder where the synthetic images will be saved.

#### Sintaxis

1. The `fit` method trains the model on the tabular data and prepares it for image generation.
    ```python
      model.fit(data)
    ```

2. The `transform` method generates and saves synthetic images in a specified folder. It requires the model to be fitted first.
    ```python
      model.transform(data, folder)
    ```

3. The `fit_transform` method combines the training and image generation steps. It fits the model to the data and generates synthetic images in one step.

  ```python
    model.fit_transform(data, folder)
  ```

#### Notes:
- **The model must be fitted** before using the `transform` method. If the model isn't fitted, a `RuntimeError` will be raised.

---

## 📚 Documentation

For detailed usage, examples, and tutorials, visit the **[TINTOlib Documentation](https://tintolib.readthedocs.io/en/latest/)**.

### How to use TINTOlib - Google Colab crash course
To get started with **TINTOlib**, a dedicated **[crash course repository](https://github.com/oeg-upm/TINTOlib-Crash_Course)** is available. It includes videoturials, slides and Jupyter Notebooks that demonstrate how to apply state-of-the-art vision models like Vision Transformers (ViTs), Convolutional Neural Networks (CNNs) and Hybrid Neural Networks to problems.
<!--It includes:

- **Slides and Jupyter notebooks** demonstrating how to:
  - Transform tabular data into images using **TINTOlib**.
  - Apply state-of-the-art vision models like **Vision Transformers (ViTs)** and **Convolutional Neural Networks (CNNs)** to classification and regression problems.

- Integration of **Hybrid Neural Networks (HyNNs)**, where:
  - **One branch** (MLP) processes the original tabular data.
  - **Another branch** (CNN or ViT) processes synthetic images.

This architecture leverages the strengths of both tabular and image-based data representations, enabling improved performance on complex machine learning tasks. The repository is ideal for those looking to integrate image-based deep learning techniques into tabular data workflows.
-->

### An Example to transform Tidy Data into synthetic image

For example, the following table shows a classic example of the [IRIS CSV dataset](https://archive.ics.uci.edu/ml/datasets/iris) as it should look like for the run:


| sepal length | sepal width | petal length | petal width | target |
|--------------|-------------|--------------|-------------|--------|
| 4.9 | 3.0 | 1.4 | 0.2 | 1 |
| 7.0 | 3.2 | 4.7 | 1.4 | 2 |
| 6.3 | 3.3 | 6.0 | 2.5 | 3 |


- The following example shows how to execute TINTOlib using the TINTO method and then display the synthetic image generated for the first row:

    ```python
      model = TINTO(problem="supervised", pixel=30, algorithm="t-SNE", steps=5, blur=True)
      model.fit_transform("iris.csv", "synthetic_images")
    ```

<!--
- **Blurring (-B)**: Create the images with blurring technique.
- **Dimensional Reduction Algorithm (-alg)**: t-SNE is used.
- **Blurring option (-oB)**: Create de images with maximum value of overlaping pixel
- **Image size (-px)**: 30x30 pixels
- **Blurring steps (-sB)**: Expand 5 pixels the blurring.
-->
<div>
<p align = "center">
<kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/blurring.png" alt="TINTO blurring" width="250"></kbd>
</p>
</div>

---

## 💬 More information

- For more detailed information, refer to the **[TINTOlib ReadTheDocs](https://tintolib.readthedocs.io/en/latest/)**.  
- GitHub repository: **[TINTOlib Documentation](https://github.com/oeg-upm/TINTOlib-Documentation)**.
- PyPI: **[PyPI](https://pypi.org/project/TINTOlib/)**.

---

## 🛡️ License

TINTOlib is available under the **[Apache License 2.0](https://github.com/oeg-upm/TINTOlib-Documentation/blob/main/LICENSE)**.

## 👥 Authors
- **[Manuel Castillo-Cara](https://github.com/manwestc)**
- **[Raúl García-Castro](https://github.com/rgcmme)**
- **[Borja Reinoso](https://github.com/borjarei) -[borjareinoso@gmail.com](borjareinoso@gmail.com)**
- **[David González Fernández](https://github.com/DavidGonzalezFernandez)**
- **[Jiayun Liu](https://github.com/DCY1117)**


## 🏛️ Contributors

<div>
<p align = "center">
<kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-oeg.png" alt="Ontology Engineering Group" width="150"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-upm.png" alt="Universidad Politécnica de Madrid" width="150"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-uned-.jpg" alt="Universidad Nacional de Educación a Distancia" width="231"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-uclm.png" alt="Universidad de Castilla-La Mancha" width="115"></kbd> 
</p>
</div>
