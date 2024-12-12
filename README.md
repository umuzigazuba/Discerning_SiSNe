# Discerning strongly interacting supernovae (siSNe)
Identifying potentially misclassified SNe Ia-CSM in a sample of SNe IIn by grouping SN light curves features based on similarity using various machine learning methods.

## Author: 

Erin Umuzigazuba (erin.umuzigazuba@gmail.com)

This code was created for my second-year master's research project in Astronomy at Leiden University, which was conducted under the supervision of Dr. María Arias. 

## General Concepts:

This repository fits the light curves of ZTF and ATLAS strongly interacting SNe. Using the best-fit parameters and light curves, features are extracted for each SNe, which are used as inputs for machine learning algorithms. Considering the high dimensionality of the datasets, the dimensionality is reduced using PCA or an autoencoder. The (reduced) parameter space is divided into clusters based on similarity using the K-means clustering algorithm.

The results can be used to identify potentially misclassified SNe Ia-CSM, by filtering SNe IIn that are grouped with SNe Ia-CSM. 

## Setup & Installation:

The repository can only be downloaded from GitHub. Download the required packages by navigating to the directory in which you downloaded the repository (make sure that `Discerning_SiSNe/` is in the working directory) and run 

```
pip install -r requirements.txt
```

**Important**: The installation has only been tested on Windows (using Windows Subsystem for Linux or WSL).

## Documentation:

Each function includes a docstring, which can be accessed through the help() function in Python.

## Workflow:

The raw data of all ZTF and ATLAS SNe Ia-CSM and IIn reported to the Transient Name Server before September 30th, 2024 are saved in the `data/raw/` folder. 

- Run the `src/data_processing.py` script to process the raw SN data. The script removes noisy/bad data points, adjusts the baseline, and accounts for Milky Way extinction and time dilation. The script saves the processed data and saves the light curve plots.

- Run the `src/parameter_estimation.py` script to fit the processed SN data to a light curve model. The data of all SNe are fit by the one-peak model from Superphot+. If a light curve shows additional peaks, the data is also fit by a two-peak model, which is the sum of the one-peak model and a Gaussian centred around the second-highest peak. The script saves the best-fit parameters and saves the best-fit light curve plots.

- Run the `src/parameter_reduction.py` script to create the datasets used as input for the machine learning algorithms. The script creates a dataset of the parameters of the model that best fits each light curve and a dataset of the one-peak model parameters only. It also creates a dataset of various properties of each light curve (peak magnitude, magnitude difference X days after the peak, duration above X % of peak flux). 

- Run the `src/principal_component_analysis.py` script to reduce the dimensionality of the datasets using principle component analysis (PCA) and apply K-means clustering on the parameter space. The script plots the dataset and clusters in two dimensions using PCA and saves the plot. For the combined dataset of one-peak model parameters and light curve properties, the script computes and plots the light curve template and saves the clusters and SNe that belong to each cluster in a file. 

- Run the `src/autoencoder.py` script to reduce the dimensionality of the datasets using an autoencoder and apply K-means clustering on the parameter space. The script plots the dataset and clusters in two dimensions using PCA and saves the plot. For the combined dataset of one-peak model parameters and light curve properties, the script computes and plots the light curve template and saves the clusters and SNe that belong to each cluster in a file. 

- Run the `src/make_plots_report.py` script to create a figure explaining the influence of each parameter of the one-peak model, which was featured in the report of the research project.

## Example:

An example fitting and extracting features from the light curve of a SN will be added in the future (see todo)
