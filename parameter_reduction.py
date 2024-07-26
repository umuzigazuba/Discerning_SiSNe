# %%

from data_processing import load_ztf_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks

# from sklearn.preprocessing import LabelEncoder
# import seaborn as sn
# from sklearn import decomposition, metrics
# from sklearn.manifold import TSNE
# import umap.umap_ as umap
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
# import random
# from scipy.stats import truncnorm, uniform, norm
# import pandas as pd
# import json
import os
import csv

plt.rcParams["text.usetex"] = True

# %%

# Load light curve data points 
ztf_id_sn_Ia_CSM = np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn = np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")
ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))

# %%

# Retrieve parameters 

def retrieve_parameters(SN_names, survey):

    parameters_OP = []
    parameters_TP = []

    for SN_id in SN_names:

        if os.path.isfile(f"Data/Analytical_parameters/{survey}/one_peak/{SN_id}_OP.npy"):
            data = np.load(f"Data/Analytical_parameters/{survey}/one_peak/{SN_id}_OP.npy")
            parameters_OP.append([SN_id, *data])

        else:
            data = np.load(f"Data/Analytical_parameters/{survey}/two_peaks/{SN_id}_TP.npy")
            parameters_TP.append([SN_id, *data])

    parameters_OP = np.array(parameters_OP, dtype = object)
    parameters_TP = np.array(parameters_TP, dtype = object)

    return parameters_OP, parameters_TP

# %%

# Retrieve reduced chi squared values

def retrieve_red_chi_squared(file_name, SN_names_Ia, SN_names_II):

    red_chi_squared_values_Ia = []
    red_chi_squared_values_II = []

    with open(file_name, "r") as file:

        reader = csv.reader(file)

        # Skip the header
        next(reader)

        for row in reader:
            
            if np.any(SN_names_Ia == row[0]):
                red_chi_squared_values_Ia.append(row)
            
            elif np.any(SN_names_II == row[0]):
                red_chi_squared_values_II.append(row)

    # Convert to NumPy array
    red_chi_squared_values_Ia = np.array(red_chi_squared_values_Ia, dtype = object)
    red_chi_squared_values_II = np.array(red_chi_squared_values_II, dtype = object)

    return red_chi_squared_values_Ia, red_chi_squared_values_II

# %%

def transform_data(SN_id, survey, parameter_values):

    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        # Load the data
        time, flux, fluxerr, filters = load_ztf_data(SN_id)

    f1_values = np.where(filters == f1)
    f2_values = np.where(filters == f2)

    # Shift the light curve so that the main peak is at time = 0 MJD
    if f1 in filters:
        peak_main_idx = np.argmax(flux[f1_values])
        peak_flux = np.copy(flux[peak_main_idx])

    else: 
        # If there is only data in the g-band, the g-band becomes the main band 
        peak_main_idx = np.argmax(flux[f2_values])
        peak_flux = np.copy(flux[peak_main_idx])
        
    transformed_values = [peak_flux * 10 ** parameter_values[0], parameter_values[1], 10 ** parameter_values[2], 10 ** parameter_values[3], parameter_values[4], 10 **parameter_values[5], 10 **parameter_values[6]] \
                        +[peak_flux * 10 ** (parameter_values[0] + parameter_values[7]), parameter_values[1] + parameter_values[8], 10 ** (parameter_values[2] + parameter_values[9]), 10 ** (parameter_values[3] + parameter_values[10]), parameter_values[4] * (10 ** parameter_values[11]), 10 ** (parameter_values[5] + parameter_values[12]), 10 ** (parameter_values[6] + parameter_values[13])]

    return transformed_values

# %%

def plot_red_chi_squared(red_chi_squared_Ia, red_chi_squared_II, survey):

    plt.hist(red_chi_squared_Ia, bins = 30, color = "tab:orange", label = "SNe Ia-CSM", zorder = 10)
    plt.hist(red_chi_squared_II, bins = 30, color = "tab:blue", label = "SNe IIn", zorder = 5)

    plt.xlabel(r"$\mathrm{X}^{2}_{red}$", fontsize = 13)
    plt.ylabel("N", fontsize = 13)
    plt.title(r"$\mathrm{X}^{2}_{red}$" + f" distribution of {survey} SNe.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

def plot_correlation(parameter_values_Ia, parameter_values_II, survey, filter, parameters):

    for idx_1 in range(len(parameters) - 1):
        for idx_2 in range(idx_1 + 1, len(parameters)): 

            plt.scatter(parameter_values_Ia[:, idx_1], parameter_values_Ia[:, idx_2], c = "tab:orange", label = "SNe Ia-CSM", zorder = 10)
            plt.scatter(parameter_values_II[:, idx_1], parameter_values_II[:, idx_2], c = "tab:blue", label = "SNe IIn", zorder = 5)

            plt.xlabel(parameters[idx_1], fontsize = 13)
            plt.ylabel(parameters[idx_2], fontsize = 13)
            plt.title(f"Parameter correlation of {survey} SNe in the {filter}-filter.")
            plt.grid(alpha = 0.3)
            plt.legend()
            plt.show()

# %%

def plot_distribution(parameter_values_Ia, parameter_values_II, survey, filter, parameters):

    for idx_1 in range(7):
        plt.hist(parameter_values_Ia[idx_1], bins = 30, color = "tab:orange", label = "SNe Ia-CSM", zorder = 10)
        plt.hist(parameter_values_II[idx_1], bins = 30, color = "tab:blue", label = "SNe IIn", zorder = 5)

        plt.xlabel(parameters[idx_1], fontsize = 13)
        plt.ylabel("N", fontsize = 13)
        plt.title(f"Parameter distribution of {survey} SNe in the {filter}-filter.")
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

# %%

parameters = ["$\mathrm{A}$", "$\mathrm{t_{0}}$", "$\mathrm{t_{rise}}$", "$\mathrm{\gamma}$", r"$\mathrm{\beta}$", "$\mathrm{t_{fall}}$"]

parameters_OP_Ia, parameters_TP_Ia = retrieve_parameters(ztf_id_sn_Ia_CSM, "ZTF")
parameters_OP_II, parameters_TP_II = retrieve_parameters(ztf_id_sn_IIn, "ZTF")

parameters_one_peak_Ia = np.concatenate((parameters_OP_Ia, parameters_TP_Ia[:, :15]))
parameters_one_peak_II = np.concatenate((parameters_OP_II, parameters_TP_II[:, :15]))

red_chi_squared_values_OP_Ia, red_chi_squared_values_OP_II = retrieve_red_chi_squared("red_chi_squared_OP.csv", ztf_id_sn_Ia_CSM, ztf_id_sn_IIn)
red_chi_squared_values_TP_Ia, red_chi_squared_values_TP_II = retrieve_red_chi_squared("red_chi_squared_TP.csv", ztf_id_sn_Ia_CSM, ztf_id_sn_IIn)

red_chi_squared_values_Ia = np.concatenate((red_chi_squared_values_OP_Ia, red_chi_squared_values_TP_Ia))
red_chi_squared_values_II = np.concatenate((red_chi_squared_values_OP_II, red_chi_squared_values_TP_II))

# %%

plot_red_chi_squared(red_chi_squared_values_Ia, red_chi_squared_values_II, "ZTF")

# %%

plot_correlation(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_correlation(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

plot_distribution(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_distribution(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

transformed_parameters_one_peak_Ia = np.array([transform_data(parameters_one_peak_Ia[idx, 0], "ZTF", parameters_one_peak_Ia[idx, 1:]) for idx in range(len(parameters_one_peak_Ia))])
transformed_parameters_one_peak_II = np.array([transform_data(parameters_one_peak_II[idx, 0], "ZTF", parameters_one_peak_II[idx, 1:]) for idx in range(len(parameters_one_peak_II))])

# %%

plot_correlation(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_correlation(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

plot_distribution(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_distribution(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

# PCA

def plot_PCA(value, label, feature):

    pca = decomposition.PCA(n_components = 2)

    pca_data = pca.fit_transform(value)

    pca_df = pd.DataFrame(data = pca_data, columns = ('Dimension 1', 'Dimension 2'))
    pca_df['SN_type'] = label

    plt.figure(figsize=(8, 6))

    colors = ['tab:orange', 'tab:blue']
    unique_labels = np.unique(label)
    for i, lbl in enumerate(unique_labels):

        class_data = pca_df[pca_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"PCA plot for feature {feature}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

# tSNE

def plot_tSNE(value, label, feature):

    model = TSNE(n_components = 2, random_state = 0)

    tsne_data = model.fit_transform(value)
    tsne_df = pd.DataFrame(data = tsne_data, columns = ('Dimension 1', 'Dimension 2'))
    tsne_df['SN_type'] = label

    plt.figure(figsize=(8, 6))

    colors = ['tab:orange', 'tab:blue']
    unique_labels = np.unique(label)
    for i, lbl in enumerate(unique_labels):

        class_data = tsne_df[tsne_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"t-SNE plot for feature {feature}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

# UMAP

def plot_UMAP(value, label, feature):

    reducer = umap.UMAP()

    UMAP_data = reducer.fit_transform(value)

    UMAP_df = pd.DataFrame(data = UMAP_data, columns = ('Dimension 1', 'Dimension 2'))
    UMAP_df['SN_type'] = label

    plt.figure(figsize=(8, 6))

    colors = ['tab:orange', 'tab:blue']
    unique_labels = np.unique(label)
    for i, lbl in enumerate(unique_labels):

        class_data = UMAP_df[UMAP_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"UMAP plot for feature {feature}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()
    
# %%
plot_PCA(ztf_parameters[:, 2:6], plot_ztf_types_name, "ZTF SNe")
plot_tSNE(ztf_parameters[:, 2:6], plot_ztf_types_name, "ZTF SNe")
plot_UMAP(ztf_parameters[:, 2:6], plot_ztf_types_name, "ZTF SNe")

# %%

np.shape(ztf_parameters[:, :7])
# %%
# %%
