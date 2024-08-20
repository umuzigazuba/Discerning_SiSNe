# %%

from data_processing import load_ztf_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks

# from sklearn.preprocessing import LabelEncoder
from sklearn import decomposition, metrics
from sklearn.manifold import TSNE
import umap.umap_ as umap
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
import os
import csv

plt.rcParams["text.usetex"] = True

np.random.seed(2804)

# %%

# Load light curve data points 
ztf_id_sn_Ia_CSM = np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn = np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")

ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))
ztf_label = np.array(["SN Ia CSM"] * len(ztf_id_sn_Ia_CSM) + ["SN IIn"] * len(ztf_id_sn_IIn))

####################################################################
# %%

# Retrieve parameters 

def retrieve_parameters(SN_names, survey):

    parameters_OP = []
    parameters_TP = []

    for SN_id in SN_names:

        if os.path.isfile(f"Data/Analytical_parameters/{survey}/one_peak/parameters_{SN_id}_OP.npy"):
            data = np.load(f"Data/Analytical_parameters/{survey}/one_peak/parameters_{SN_id}_OP.npy")
            parameters_OP.append([SN_id, *data])

        else:
            data = np.load(f"Data/Analytical_parameters/{survey}/two_peaks/parameters_{SN_id}_TP.npy")
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
        
            name = row[0]
            values = list(map(float, row[1:]))  

            if name in SN_names_Ia:
                red_chi_squared_values_Ia.append([name] + values)

            elif name in SN_names_II:
                red_chi_squared_values_II.append([name] + values)

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

    min_bin = np.min(np.concatenate((red_chi_squared_Ia, red_chi_squared_II)))
    max_bin = np.max(np.concatenate((red_chi_squared_Ia, red_chi_squared_II)))
    bins = np.linspace(min_bin, max_bin, 25)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "bar", alpha = 0.4, zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "bar", alpha = 0.4, zorder = 5)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "step",  fill = False, label = "SNe IIn", zorder = 5)

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

def plot_correlation_contour(parameter_values_Ia, parameter_values_II, survey, filter, parameters):

    for idx_1 in range(len(parameters) - 1):
        for idx_2 in range(idx_1 + 1, len(parameters)): 

            sn.set_style("white")
            sn.kdeplot(x = parameter_values_Ia[:, idx_1], y = parameter_values_Ia[:, idx_2], thresh = 0.3, color = "tab:orange", label = "SNe Ia-CSM", zorder = 10)
            sn.kdeplot(x = parameter_values_II[:, idx_1], y = parameter_values_II[:, idx_2], thresh = 0.3, color = "tab:blue", label = "SNe IIn", zorder = 5)

            plt.xlabel(parameters[idx_1], fontsize = 13)
            plt.ylabel(parameters[idx_2], fontsize = 13)
            plt.title(f"Parameter correlation of {survey} SNe in the {filter}-filter.")
            plt.grid(alpha = 0.3)
            plt.legend()
            plt.show()

# %%

def plot_distribution(parameter_values_Ia, parameter_values_II, survey, filter, parameters):

    for idx_1 in range(len(parameters)):

        min_bin = np.min(np.concatenate((parameter_values_Ia[:, idx_1], parameter_values_II[:, idx_1])))
        max_bin = np.max(np.concatenate((parameter_values_Ia[:, idx_1], parameter_values_II[:, idx_1])))
        bins = np.linspace(min_bin, max_bin, 25)

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, density = True, linewidth = 2, color = "tab:orange", histtype = "bar", alpha = 0.4, zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, density = True, linewidth = 2, color = "tab:blue", histtype = "bar", alpha = 0.4, zorder = 5)

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, density = True, linewidth = 2, color = "tab:orange", histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, density = True, linewidth = 2, color = "tab:blue", histtype = "step", fill = False, label = "SNe IIn", zorder = 5)

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
parameters_one_peak = np.concatenate((parameters_one_peak_Ia, parameters_one_peak_II))

red_chi_squared_values_OP_Ia, red_chi_squared_values_OP_II = retrieve_red_chi_squared("Data/Analytical_parameters/ZTF/one_peak/red_chi_squared_OP.csv", ztf_id_sn_Ia_CSM, ztf_id_sn_IIn)
red_chi_squared_values_TP_Ia, red_chi_squared_values_TP_II = retrieve_red_chi_squared("Data/Analytical_parameters/ZTF/two_peaks/red_chi_squared_TP.csv", ztf_id_sn_Ia_CSM, ztf_id_sn_IIn)

red_chi_squared_values_Ia = np.concatenate((red_chi_squared_values_OP_Ia, red_chi_squared_values_TP_Ia))
red_chi_squared_values_II = np.concatenate((red_chi_squared_values_OP_II, red_chi_squared_values_TP_II))

# %%

plot_red_chi_squared(red_chi_squared_values_Ia[:, 1], red_chi_squared_values_II[:, 1], "ZTF")

# %%

plot_correlation(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_correlation(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

plot_correlation_contour(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_correlation_contour(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

plot_distribution(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
plot_distribution(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# %%

# transformed_parameters_one_peak_Ia = np.array([transform_data(parameters_one_peak_Ia[idx, 0], "ZTF", parameters_one_peak_Ia[idx, 1:]) for idx in range(len(parameters_one_peak_Ia))])
# transformed_parameters_one_peak_II = np.array([transform_data(parameters_one_peak_II[idx, 0], "ZTF", parameters_one_peak_II[idx, 1:]) for idx in range(len(parameters_one_peak_II))])

# # %%

# plot_correlation(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
# plot_correlation(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

# # %%

# plot_distribution(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], "ZTF", "r", parameters)
# plot_distribution(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], "ZTF", "g", parameters)

####################################################################
# %%

def retrieve_global_parameters(SN_id, survey, peak_number, parameter_values):

    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        time, flux, _, filters = load_ztf_data(SN_id)

    f1_values = np.where(filters == f1)
    f2_values = np.where(filters == f2)

    # Shift the light curve so that the main peak is at time = 0 MJD
    if f1 in filters:
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

    else: 
        # If there is only data in the g-band, the g-band becomes the main band 
        peak_main_idx = np.argmax(flux[f2_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

    time -= peak_time

    amount_fit = 200
    time_fit = np.concatenate((np.linspace(time.min(), time.max(), amount_fit), np.linspace(time.min(), time.max(), amount_fit)))
    f1_values_fit = np.arange(amount_fit)
    f2_values_fit = np.arange(amount_fit) + amount_fit

    if peak_number == 1:
        flux_fit = light_curve_one_peak(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    elif peak_number == 2:
        flux_fit = light_curve_two_peaks(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    if len(f1_values) != 0:
        peak_flux_f1 = np.max(flux_fit[f1_values_fit])

    else:
        peak_flux_f1 = np.nan

    if len(f2_values) != 0:
        peak_flux_f2 = np.max(flux_fit[f2_values_fit])

    else:
        peak_flux_f2 = np.nan

    return np.array([peak_flux_f1, peak_flux_f2])
            

# %%

peak_flux_OP_Ia = np.empty((0,2))

for id, SN_id in enumerate(parameters_OP_Ia[:, 0]):

    peak_flux = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_OP_Ia[id, 1:])
    peak_flux_OP_Ia = np.append(peak_flux_OP_Ia, [peak_flux], axis = 0)

peak_flux_OP_II = np.empty((0,2))

for id, SN_id in enumerate(parameters_OP_II[:, 0]):

    peak_flux = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_OP_II[id, 1:])
    peak_flux_OP_II = np.append(peak_flux_OP_II, [peak_flux], axis = 0)

peak_flux_TP_Ia = np.empty((0,2))

for id, SN_id in enumerate(parameters_TP_Ia[:, 0]):

    peak_flux = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_TP_Ia[id, 1:])
    peak_flux_TP_Ia = np.append(peak_flux_TP_Ia, [peak_flux], axis = 0)

peak_flux_TP_II = np.empty((0,2))

for id, SN_id in enumerate(parameters_TP_II[:, 0]):

    peak_flux = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_TP_II[id, 1:])
    peak_flux_TP_II = np.append(peak_flux_TP_II, [peak_flux], axis = 0)

peak_flux_Ia = np.concatenate((peak_flux_OP_Ia, peak_flux_TP_Ia))
peak_flux_II = np.concatenate((peak_flux_OP_II, peak_flux_TP_II))
peak_flux = np.concatenate((peak_flux_Ia, peak_flux_II))

# %%

def plot_global_parameter(parameter_values_Ia, parameter_values_II, parameter_label, parameter_name, survey, filter):

    min_bin = np.min(np.concatenate((parameter_values_Ia, parameter_values_II)))
    max_bin = np.max(np.concatenate((parameter_values_Ia, parameter_values_II)))
    bins = np.linspace(min_bin, max_bin, 25)

    plt.hist(parameter_values_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "bar", alpha = 0.4, zorder = 10)
    plt.hist(parameter_values_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "bar", alpha = 0.4, zorder = 5)

    plt.hist(parameter_values_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
    plt.hist(parameter_values_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "step",  fill = False, label = "SNe IIn", zorder = 5)

    plt.xlabel(f"{parameter_label}", fontsize = 13)
    plt.ylabel("N", fontsize = 13)
    plt.title(f" {parameter_name} distribution of {survey} SNe in the {filter}-filter.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

plot_global_parameter(peak_flux_Ia[:, 0], peak_flux_II[:, 0], "Flux $(\mu Jy)$", "Peak flux", "ZTF", "r")
plot_global_parameter(peak_flux_Ia[:, 1], peak_flux_II[:, 1], "Flux $(\mu Jy)$", "Peak flux", "ZTF", "g")

####################################################################
# %%

# PCA

def plot_PCA(parameter_values, SN_type):

    pca = decomposition.PCA(n_components = 2)

    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data, columns = ('Dimension 1', 'Dimension 2'))
    pca_df['SN_type'] = SN_type

    plt.figure(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange']
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = pca_df[pca_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"PCA plot")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

# tSNE

def plot_tSNE(parameter_values, SN_type):

    model = TSNE(n_components = 2, random_state = 2804)

    tsne_data = model.fit_transform(parameter_values)
    tsne_df = pd.DataFrame(data = tsne_data, columns = ('Dimension 1', 'Dimension 2'))
    tsne_df['SN_type'] = SN_type

    plt.figure(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange']
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = tsne_df[tsne_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"t-SNE plot")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

# UMAP

def plot_UMAP(parameter_values, SN_type):

    reducer = umap.UMAP(random_state = 2804)

    UMAP_data = reducer.fit_transform(parameter_values)

    UMAP_df = pd.DataFrame(data = UMAP_data, columns = ('Dimension 1', 'Dimension 2'))
    UMAP_df['SN_type'] = SN_type

    plt.figure(figsize=(8, 6))

    colors = ['tab:blue', 'tab:orange']
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = UMAP_df[UMAP_df['SN_type'] == lbl]
        plt.scatter(class_data['Dimension 1'], class_data['Dimension 2'], c = colors[i], label = lbl)

    plt.title(f"UMAP plot")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()
    
# %%

plot_PCA(parameters_one_peak[:, 1:], ztf_label)

plot_tSNE(parameters_one_peak[:, 1:], ztf_label)

plot_UMAP(parameters_one_peak[:, 1:], ztf_label)

# %%
