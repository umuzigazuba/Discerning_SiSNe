# %%

from data_processing import load_ztf_data, load_atlas_data, ztf_micro_flux_to_magnitude
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

# %%

# Load light curve data points 
ztf_id_sn_Ia_CSM = np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn = np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")

ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))
ztf_label = np.array(["SN Ia CSM"] * len(ztf_id_sn_Ia_CSM) + ["SN IIn"] * len(ztf_id_sn_IIn))


atlas_id_sn_Ia_CSM = np.loadtxt("Data/ATLAS_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
atlas_id_sn_IIn = np.loadtxt("Data/ATLAS_ID_SNe_IIn", delimiter = ",", dtype = "str")

atlas_id = np.concatenate((atlas_id_sn_Ia_CSM, atlas_id_sn_IIn))
atlas_label = np.array(["SN Ia CSM"] * len(atlas_id_sn_Ia_CSM) + ["SN IIn"] * len(atlas_id_sn_IIn))

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

    elif survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, fluxerr, filters = load_atlas_data(SN_id)

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
                       + [peak_flux * 10 ** (parameter_values[0] + parameter_values[7]), parameter_values[1] + parameter_values[8], 10 ** (parameter_values[2] + parameter_values[9]), 10 ** (parameter_values[3] + parameter_values[10]), parameter_values[4] * (10 ** parameter_values[11]), 10 ** (parameter_values[5] + parameter_values[12]), 10 ** (parameter_values[6] + parameter_values[13])]

    return transformed_values

# %%

def plot_red_chi_squared(red_chi_squared_Ia, red_chi_squared_II, percentile_95, survey):

    min_bin = np.min(np.concatenate((red_chi_squared_Ia, red_chi_squared_II)))
    max_bin = np.max(np.concatenate((red_chi_squared_Ia, red_chi_squared_II)))
    bins = np.linspace(min_bin, max_bin, 25)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "bar", alpha = 0.4, zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "bar", alpha = 0.4, zorder = 5)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = "tab:orange", histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = "tab:blue", histtype = "step",  fill = False, label = "SNe IIn", zorder = 5)

    plt.axvline(x = percentile_95, color = "black", linestyle = "dashed", label = "95th percentile")

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

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, linewidth = 2, color = "tab:orange", histtype = "bar", alpha = 0.4, zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, linewidth = 2, color = "tab:blue", histtype = "bar", alpha = 0.4, zorder = 5)

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, linewidth = 2, color = "tab:orange", histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, linewidth = 2, color = "tab:blue", histtype = "step", fill = False, label = "SNe IIn", zorder = 5)

        plt.xlabel(parameters[idx_1], fontsize = 13)
        plt.ylabel("N", fontsize = 13)
        plt.title(f"Parameter distribution of {survey} SNe in the {filter}-filter.")
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

# %%

survey = "ZTF"

if survey == "ZTF":
    SN_names_Ia = ztf_id_sn_Ia_CSM
    SN_names_II = ztf_id_sn_IIn

    f1 = "r"
    f2 = "g"

elif survey == "ATLAS":
    SN_names_Ia = atlas_id_sn_Ia_CSM
    SN_names_II = atlas_id_sn_IIn

    f1 = "o"
    f2 = "c"

red_chi_squared_values_OP_Ia, red_chi_squared_values_OP_II = retrieve_red_chi_squared(f"Data/Analytical_parameters/{survey}/one_peak/red_chi_squared_OP.csv", SN_names_Ia, SN_names_II)
red_chi_squared_values_TP_Ia, red_chi_squared_values_TP_II = retrieve_red_chi_squared(f"Data/Analytical_parameters/{survey}/two_peaks/red_chi_squared_TP.csv", SN_names_Ia, SN_names_II)

red_chi_squared_values_Ia = np.concatenate((red_chi_squared_values_OP_Ia, red_chi_squared_values_TP_Ia))
red_chi_squared_values_II = np.concatenate((red_chi_squared_values_OP_II, red_chi_squared_values_TP_II))

red_chi_squared_values = np.concatenate((red_chi_squared_values_Ia, red_chi_squared_values_II))
percentile_95 = np.percentile(red_chi_squared_values[:, 1], 95)

# Remove light curves with reduced chi squared larger than the 95th percentile
cut_light_curves = np.where(red_chi_squared_values[:, 1] > percentile_95)

cut_light_curves_Ia = np.where(np.isin(SN_names_Ia, red_chi_squared_values[cut_light_curves, 0][0]))[0]
cut_light_curves_II = np.where(np.isin(SN_names_II, red_chi_squared_values[cut_light_curves, 0][0]))[0]

SN_names_Ia = np.delete(SN_names_Ia, cut_light_curves_Ia)
SN_names_II = np.delete(SN_names_II, cut_light_curves_II)
# %%

plot_red_chi_squared(red_chi_squared_values_Ia[:, 1], red_chi_squared_values_II[:, 1], percentile_95, survey)

# %%

parameters = ["$\mathrm{A}$", "$\mathrm{t_{0}}$", "$\mathrm{t_{rise}}$", "$\mathrm{\gamma}$", r"$\mathrm{\beta}$", "$\mathrm{t_{fall}}$"]

parameters_OP_Ia, parameters_TP_Ia = retrieve_parameters(SN_names_Ia, survey)
parameters_OP_II, parameters_TP_II = retrieve_parameters(SN_names_II, survey)

parameters_one_peak_Ia = np.concatenate((parameters_OP_Ia, parameters_TP_Ia[:, :15]))
parameters_one_peak_II = np.concatenate((parameters_OP_II, parameters_TP_II[:, :15]))
parameters_one_peak = np.concatenate((parameters_one_peak_Ia, parameters_one_peak_II))

# %%

plot_correlation(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], survey, f1, parameters)
plot_correlation(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], survey, f2, parameters)

# %%

# plot_correlation_contour(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], survey, f1, parameters)
# plot_correlation_contour(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], survey, f2, parameters)

# %%

plot_distribution(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], survey, f1, parameters)
plot_distribution(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], survey, f2, parameters)

# %%

# transformed_parameters_one_peak_Ia = np.array([transform_data(parameters_one_peak_Ia[idx, 0], survey, parameters_one_peak_Ia[idx, 1:]) for idx in range(len(parameters_one_peak_Ia))])
# transformed_parameters_one_peak_II = np.array([transform_data(parameters_one_peak_II[idx, 0], survey, parameters_one_peak_II[idx, 1:]) for idx in range(len(parameters_one_peak_II))])

# # %%

# plot_correlation(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], survey, f1, parameters)
# plot_correlation(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], survey, f2, parameters)

# # %%

# plot_distribution(transformed_parameters_one_peak_Ia[:, 1:7], transformed_parameters_one_peak_II[:, 1:7], survey, f1, parameters)
# plot_distribution(transformed_parameters_one_peak_Ia[:, 8:14], transformed_parameters_one_peak_II[:, 8:14], survey, f2, parameters)

####################################################################
# %%

def retrieve_best_fit(SN_id, survey, peak_number, parameter_values):

    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        time, flux, _, filters = load_ztf_data(SN_id)

    if survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, _, filters = load_atlas_data(SN_id)

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
    time_fit = np.concatenate((np.linspace(-150, 500, amount_fit), np.linspace(-150, 500, amount_fit)))
    f1_values_fit = np.arange(amount_fit)
    f2_values_fit = np.arange(amount_fit) + amount_fit

    if peak_number == 1:
        flux_fit = light_curve_one_peak(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    elif peak_number == 2:
        flux_fit = light_curve_two_peaks(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    magnitude_fit = ztf_micro_flux_to_magnitude(flux_fit)

    time_fit += peak_time

    return time_fit, flux_fit, magnitude_fit

# %%

retrieve_best_fit(parameters_one_peak_II[40, 0], survey, 1, parameters_one_peak_II[40, 1:])

# %%

    if len(f1_values) != 0:
        peak_fit_idx_f1 = np.argmax(flux_fit[f1_values_fit])
        peak_time_fit_f1 = np.copy(time_fit[peak_fit_idx_f1])
        peak_flux_fit_f1 = np.copy(flux_fit[peak_fit_idx_f1])
        time_fit_f1 = time_fit - peak_time_fit_f1

        peak_magnitude_f1 = magnitude_fit[peak_fit_idx_f1]

        thirty_days_after_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] - 30))
        thirty_days_magnitude_f1 = magnitude_fit[thirty_days_after_peak_f1]
        thirty_days_magnitude_difference_f1 = peak_magnitude_f1 - thirty_days_magnitude_f1

        half_peak_flux_f1 = np.argmin(np.abs(flux_fit[f1_values_fit][peak_fit_idx_f1:] - peak_flux_fit_f1 * 0.5)) + peak_fit_idx_f1
        half_peak_time_f1 = time_fit_f1[half_peak_flux_f1]

        fifth_peak_flux_f1 = np.argmin(np.abs(flux_fit[f1_values_fit][peak_fit_idx_f1:] - peak_flux_fit_f1 * 0.2)) + peak_fit_idx_f1
        fifth_peak_time_f1 = time_fit_f1[fifth_peak_flux_f1]

    else:
        peak_magnitude_f1 = np.nan
        thirty_days_magnitude_difference_f1 = np.nan
        half_peak_time_f1 = np.nan
        fifth_peak_time_f1 = np.nan

    if len(f2_values) != 0:
        peak_fit_idx_f2 = np.argmax(flux_fit[f2_values_fit]) + amount_fit
        peak_time_fit_f2 = np.copy(time_fit[peak_fit_idx_f2])
        peak_flux_fit_f2 = np.copy(flux_fit[peak_fit_idx_f2])
        time_fit_f2 = time_fit - peak_time_fit_f2

        peak_magnitude_f2 = magnitude_fit[peak_fit_idx_f2]

        thirty_days_after_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] - 30)) + amount_fit
        thirty_days_magnitude_f2 = magnitude_fit[thirty_days_after_peak_f2]
        thirty_days_magnitude_difference_f2 = peak_magnitude_f2 - thirty_days_magnitude_f2

        half_peak_flux_f2 = np.argmin(np.abs(flux_fit[f2_values_fit][(peak_fit_idx_f2 - amount_fit):] - peak_flux_fit_f2 * 0.5)) + peak_fit_idx_f2
        half_peak_time_f2 = time_fit_f2[half_peak_flux_f2]

        fifth_peak_flux_f2 = np.argmin(np.abs(flux_fit[f2_values_fit][(peak_fit_idx_f2 - amount_fit):] - peak_flux_fit_f2 * 0.2)) + peak_fit_idx_f2
        fifth_peak_time_f2 = time_fit_f2[fifth_peak_flux_f2]

    else:
        peak_magnitude_f2 = np.nan
        thirty_days_magnitude_difference_f2 = np.nan
        half_peak_time_f2 = np.nan
        fifth_peak_time_f2 = np.nan

    return np.array([peak_magnitude_f1, peak_magnitude_f2]), \
           np.array([thirty_days_magnitude_difference_f1, thirty_days_magnitude_difference_f2]), \
           np.array([half_peak_time_f1, half_peak_time_f2]), \
           np.array([fifth_peak_time_f1, fifth_peak_time_f2])
            

# %%

peak_magnitude_OP_Ia = np.empty((0,2))
thirty_days_magnitude_difference_OP_Ia = np.empty((0,2))
half_peak_time_OP_Ia = np.empty((0,2))
fifth_peak_time_OP_Ia = np.empty((0,2))

for id, SN_id in enumerate(parameters_OP_Ia[:, 0]):

    peak_magnitude, thirty_days_magnitude_difference, half_peak_time, fifth_peak_time = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_OP_Ia[id, 1:])
    
    peak_magnitude_OP_Ia = np.append(peak_magnitude_OP_Ia, [peak_magnitude], axis = 0)
    thirty_days_magnitude_difference_OP_Ia = np.append(thirty_days_magnitude_difference_OP_Ia, [thirty_days_magnitude_difference], axis = 0)
    half_peak_time_OP_Ia = np.append(half_peak_time_OP_Ia, [half_peak_time], axis = 0)
    fifth_peak_time_OP_Ia = np.append(fifth_peak_time_OP_Ia, [fifth_peak_time], axis = 0)

peak_magnitude_OP_II = np.empty((0,2))
thirty_days_magnitude_difference_OP_II = np.empty((0,2))
half_peak_time_OP_II = np.empty((0,2))
fifth_peak_time_OP_II = np.empty((0,2))

for id, SN_id in enumerate(parameters_OP_II[:, 0]):

    peak_magnitude, thirty_days_magnitude_difference, half_peak_time, fifth_peak_time = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_OP_II[id, 1:])
    
    peak_magnitude_OP_II = np.append(peak_magnitude_OP_II, [peak_magnitude], axis = 0)
    thirty_days_magnitude_difference_OP_II = np.append(thirty_days_magnitude_difference_OP_II, [thirty_days_magnitude_difference], axis = 0)
    half_peak_time_OP_II = np.append(half_peak_time_OP_II, [half_peak_time], axis = 0)
    fifth_peak_time_OP_II = np.append(fifth_peak_time_OP_II, [fifth_peak_time], axis = 0)

peak_magnitude_TP_Ia = np.empty((0,2))
thirty_days_magnitude_difference_TP_Ia = np.empty((0,2))
half_peak_time_TP_Ia = np.empty((0,2))
fifth_peak_time_TP_Ia = np.empty((0,2))

for id, SN_id in enumerate(parameters_TP_Ia[:, 0]):

    peak_magnitude, thirty_days_magnitude_difference, half_peak_time, fifth_peak_time = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_TP_Ia[id, 1:])
    
    peak_magnitude_TP_Ia = np.append(peak_magnitude_TP_Ia, [peak_magnitude], axis = 0)
    thirty_days_magnitude_difference_TP_Ia = np.append(thirty_days_magnitude_difference_TP_Ia, [thirty_days_magnitude_difference], axis = 0)
    half_peak_time_TP_Ia = np.append(half_peak_time_TP_Ia, [half_peak_time], axis = 0)
    fifth_peak_time_TP_Ia = np.append(fifth_peak_time_TP_Ia, [fifth_peak_time], axis = 0)

peak_magnitude_TP_II = np.empty((0,2))
thirty_days_magnitude_difference_TP_II = np.empty((0,2))
half_peak_time_TP_II = np.empty((0,2))
fifth_peak_time_TP_II = np.empty((0,2))

for id, SN_id in enumerate(parameters_TP_II[:, 0]):

    peak_magnitude, thirty_days_magnitude_difference, half_peak_time, fifth_peak_time = retrieve_global_parameters(SN_id, "ZTF", 1, parameters_TP_II[id, 1:])
    
    peak_magnitude_TP_II = np.append(peak_magnitude_TP_II, [peak_magnitude], axis = 0)
    thirty_days_magnitude_difference_TP_II = np.append(thirty_days_magnitude_difference_TP_II, [thirty_days_magnitude_difference], axis = 0)
    half_peak_time_TP_II = np.append(half_peak_time_TP_II, [half_peak_time], axis = 0)
    fifth_peak_time_TP_II = np.append(fifth_peak_time_TP_II, [fifth_peak_time], axis = 0)

peak_magnitude_Ia = np.concatenate((peak_magnitude_OP_Ia, peak_magnitude_TP_Ia))
peak_magnitude_II = np.concatenate((peak_magnitude_OP_II, peak_magnitude_TP_II))
peak_magnitude = np.concatenate((peak_magnitude_Ia, peak_magnitude_II))

thirty_days_magnitude_difference_Ia = np.concatenate((thirty_days_magnitude_difference_OP_Ia, thirty_days_magnitude_difference_TP_Ia))
thirty_days_magnitude_difference_II = np.concatenate((thirty_days_magnitude_difference_OP_II, thirty_days_magnitude_difference_TP_II))
thirty_days_magnitude_difference = np.concatenate((thirty_days_magnitude_difference_Ia, thirty_days_magnitude_difference_II))

half_peak_time_Ia = np.concatenate((half_peak_time_OP_Ia, half_peak_time_TP_Ia))
half_peak_time_II = np.concatenate((half_peak_time_OP_II, half_peak_time_TP_II))
half_peak_time = np.concatenate((half_peak_time_Ia, half_peak_time_II))

fifth_peak_time_Ia = np.concatenate((fifth_peak_time_OP_Ia, fifth_peak_time_TP_Ia))
fifth_peak_time_II = np.concatenate((fifth_peak_time_OP_II, fifth_peak_time_TP_II))
fifth_peak_time = np.concatenate((fifth_peak_time_Ia, fifth_peak_time_II))

# %%

global_parameters = np.column_stack((peak_magnitude, thirty_days_magnitude_difference, half_peak_time, fifth_peak_time))
print(np.shape(global_parameters))

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

def plot_comparisson_global_parameter(parameter_1_values_Ia, parameter_1_values_II, parameter_2_values_Ia, parameter_2_values_II, parameter_1_label, parameter_2_label, survey, filter):

    plt.scatter(parameter_1_values_Ia, parameter_2_values_Ia, c = "tab:orange", label = "SNe Ia-CSM", zorder = 10)
    plt.scatter(parameter_1_values_II, parameter_2_values_II, c = "tab:blue", label = "SNe IIn", zorder = 5)

    plt.xlabel(f"{parameter_1_label}", fontsize = 13)
    plt.ylabel(f"{parameter_2_label}", fontsize = 13)
    plt.title(f" Parameter comparisson of {survey} SNe in the {filter}-filter.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

# %%

plot_global_parameter(peak_magnitude_Ia[:, 0], peak_magnitude_II[:, 0], "Absolute magnitude", "Peak absolute magnitude", "ZTF", "r")
plot_global_parameter(peak_magnitude_Ia[:, 1], peak_magnitude_II[:, 1], "Absolute magnitude", "Peak absolute magnitude", "ZTF", "g")

plot_global_parameter(thirty_days_magnitude_difference_Ia[:, 0], thirty_days_magnitude_difference_II[:, 0], "Absolute magnitude", "Thirty days absolute magnitude difference", "ZTF", "r")
plot_global_parameter(thirty_days_magnitude_difference_Ia[:, 1], thirty_days_magnitude_difference_II[:, 1], "Absolute magnitude", "Thirty days absolute magnitude difference", "ZTF", "g")

plot_global_parameter(half_peak_time_Ia[:, 0], half_peak_time_II[:, 0], "Time (days)", "Time above half of peak flux", "ZTF", "r")
plot_global_parameter(half_peak_time_Ia[:, 1], half_peak_time_II[:, 1], "Time (days)", "Time above half of peak flux", "ZTF", "g")

plot_global_parameter(fifth_peak_time_Ia[:, 0], fifth_peak_time_II[:, 0], "Time (days)", "Time above fifth of peak flux", "ZTF", "r")
plot_global_parameter(fifth_peak_time_Ia[:, 1], fifth_peak_time_II[:, 1], "Time (days)", "Time above fifth of peak flux", "ZTF", "g")

# plot_comparisson_global_parameter(half_peak_time_Ia[:, 0], half_peak_time_II[:, 0], \
#                                   peak_magnitude_Ia[:, 0], peak_magnitude_II[:, 0], \
#                                   "Time above half of peak flux (days)", "Thirty days absolute magnitude difference", "ZTF", "r")
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

plot_PCA(global_parameters, ztf_label)

plot_tSNE(global_parameters, ztf_label)

plot_UMAP(global_parameters, ztf_label)

# %%
global_parameters
# %%

