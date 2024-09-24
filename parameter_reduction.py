# %%

from data_processing import load_ztf_data, load_atlas_data, ztf_micro_flux_to_magnitude
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, metrics
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.integrate import quad
import seaborn as sn
import pandas as pd
import os
import csv
import warnings

plt.rcParams["text.usetex"] = True

# %%

# Load light curve data points 
ztf_id_sn_Ia_CSM = np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn = np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")

atlas_id_sn_Ia_CSM = np.loadtxt("Data/ATLAS_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
atlas_id_sn_IIn = np.loadtxt("Data/ATLAS_ID_SNe_IIn", delimiter = ",", dtype = "str")

# %% 

def delete_single_filter_SN(SN_names, survey):

    to_delete = np.array([])

    if survey == "ZTF":
        for id, SN_id in enumerate(SN_names):

            _, _, _, filters = load_ztf_data(SN_id)

            if "g" not in filters or "r" not in filters:
                to_delete = np.append(to_delete, id)

    if survey == "ATLAS":
        for id, SN_id in enumerate(SN_names):

            _, _, _, filters = load_atlas_data(SN_id)

            if "o" not in filters or "c" not in filters:
                to_delete = np.append(to_delete, id)

    SN_names = np.delete(SN_names, to_delete.astype(int))

    return SN_names

# %%

ztf_id_sn_Ia_CSM = delete_single_filter_SN(ztf_id_sn_Ia_CSM, "ZTF")
ztf_id_sn_IIn = delete_single_filter_SN(ztf_id_sn_IIn, "ZTF")

atlas_id_sn_Ia_CSM = delete_single_filter_SN(atlas_id_sn_Ia_CSM, "ATLAS")
atlas_id_sn_IIn = delete_single_filter_SN(atlas_id_sn_IIn, "ATLAS")

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

    plt.axvline(x = percentile_95, color = "black", linestyle = "dashed", label = f"95th percentile = {percentile_95:.2f}")

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

SN_labels = np.array(["SN Ia CSM"] * len(SN_names_Ia) + ["SN IIn"] * len(SN_names_II))
SN_labels_color = np.array([0] * len(SN_names_Ia) + [1] * len(SN_names_II))
# %%

plot_red_chi_squared(red_chi_squared_values_Ia[:, 1], red_chi_squared_values_II[:, 1], percentile_95, survey)

# %%

parameters = ["$\mathrm{A}$", "$\mathrm{t_{0}}$", "$\mathrm{t_{rise}}$", "$\mathrm{\gamma}$", r"$\mathrm{\beta}$", "$\mathrm{t_{fall}}$"]

parameters_OP_Ia, parameters_TP_Ia = retrieve_parameters(SN_names_Ia, survey)
parameters_OP_II, parameters_TP_II = retrieve_parameters(SN_names_II, survey)

parameters_one_peak_Ia = np.concatenate((parameters_OP_Ia, parameters_TP_Ia[:, :15]))
parameters_one_peak_II = np.concatenate((parameters_OP_II, parameters_TP_II[:, :15]))
parameters_one_peak = np.concatenate((parameters_one_peak_Ia, parameters_one_peak_II))

parameters_two_peaks_Ia = np.concatenate((np.concatenate((parameters_OP_Ia, np.zeros((len(parameters_OP_Ia), 6))), axis = 1), parameters_TP_Ia))
parameters_two_peaks_II = np.concatenate((np.concatenate((parameters_OP_II, np.zeros((len(parameters_OP_II), 6))), axis = 1), parameters_TP_II))
parameters_two_peaks = np.concatenate((parameters_two_peaks_Ia, parameters_two_peaks_II))

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

def calculate_global_parameters(SN_id, survey, peak_number, parameter_values):

    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        time, flux, fluxerr, filters = load_ztf_data(SN_id)

    if survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, fluxerr, filters = load_atlas_data(SN_id)

    f1_values = np.where(filters == f1)
    f2_values = np.where(filters == f2)

    # Shift the light curve so that the main peak is at time = 0 MJD
    peak_main_idx = np.argmax(flux[f1_values])
    peak_time = np.copy(time[peak_main_idx])
    peak_flux = np.copy(flux[peak_main_idx])

    time -= peak_time

    amount_fit = 1000
    time_fit = np.concatenate((np.linspace(-150, 500, amount_fit), np.linspace(-150, 500, amount_fit)))
    f1_values_fit = np.arange(amount_fit)
    f2_values_fit = np.arange(amount_fit) + amount_fit

    if peak_number == 1:
        flux_fit = light_curve_one_peak(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    elif peak_number == 2:
        flux_fit = light_curve_two_peaks(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    magnitude_fit = ztf_micro_flux_to_magnitude(flux_fit)

    time_fit += peak_time

    # In filter 1
    peak_fit_idx_f1 = np.argmax(flux_fit[f1_values_fit])
    peak_time_fit_f1 = np.copy(time_fit[peak_fit_idx_f1])
    peak_flux_fit_f1 = np.copy(flux_fit[peak_fit_idx_f1])
    time_fit_f1 = time_fit - peak_time_fit_f1

    # Peak magnitude
    peak_magnitude_f1 = magnitude_fit[peak_fit_idx_f1]

    # Rise time 
    explosion_time_f1 = np.argmin(np.abs(flux_fit[f1_values_fit][:peak_fit_idx_f1] - 0.1))
    rise_time_f1 = time_fit_f1[peak_fit_idx_f1] - time_fit_f1[explosion_time_f1]

    # Magnitude difference peak - 10 days before
    ten_days_before_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] + 10))
    ten_days_magnitude_f1 = magnitude_fit[ten_days_before_peak_f1]
    ten_days_magnitude_difference_f1 = np.abs(peak_magnitude_f1 - ten_days_magnitude_f1)

    # Magnitude difference peak - 15 days after
    fifteen_days_after_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] - 15))
    fifteen_days_magnitude_f1 = magnitude_fit[fifteen_days_after_peak_f1]
    fifteen_days_magnitude_difference_f1 = np.abs(peak_magnitude_f1 - fifteen_days_magnitude_f1)

    # Magnitude difference peak - 30 days after
    thirty_days_after_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] - 30))
    thirty_days_magnitude_f1 = magnitude_fit[thirty_days_after_peak_f1]
    thirty_days_magnitude_difference_f1 = np.abs(peak_magnitude_f1 - thirty_days_magnitude_f1)

    # Duration above half of the peak flux
    half_peak_flux_f1 = np.argmin(np.abs(flux_fit[f1_values_fit][peak_fit_idx_f1:] - peak_flux_fit_f1 * 0.5)) + peak_fit_idx_f1
    half_peak_time_f1 = time_fit_f1[half_peak_flux_f1]

    # Duration above a fifth of the peak flux
    fifth_peak_flux_f1 = np.argmin(np.abs(flux_fit[f1_values_fit][peak_fit_idx_f1:] - peak_flux_fit_f1 * 0.2)) + peak_fit_idx_f1
    fifth_peak_time_f1 = time_fit_f1[fifth_peak_flux_f1]

    # In filter 2
    peak_fit_idx_f2 = np.argmax(flux_fit[f2_values_fit]) + amount_fit
    peak_time_fit_f2 = np.copy(time_fit[peak_fit_idx_f2])
    peak_flux_fit_f2 = np.copy(flux_fit[peak_fit_idx_f2])
    time_fit_f2 = time_fit - peak_time_fit_f2

    # Peak magnitude
    peak_magnitude_f2 = magnitude_fit[peak_fit_idx_f2]

    # Rise time 
    explosion_time_f2 = np.argmin(np.abs(flux_fit[f2_values_fit][:(peak_fit_idx_f2 - amount_fit)] - 0.1)) 
    rise_time_f2 = time_fit_f2[peak_fit_idx_f2] - time_fit_f2[explosion_time_f2]

    # Magnitude difference peak - 10 days before
    ten_days_before_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] + 10)) + amount_fit
    ten_days_magnitude_f2 = magnitude_fit[ten_days_before_peak_f2]
    ten_days_magnitude_difference_f2 = np.abs(peak_magnitude_f2 - ten_days_magnitude_f2)

    # Magnitude difference peak - 15 days after
    fifteen_days_after_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] - 15)) + amount_fit
    fifteen_days_magnitude_f2 = magnitude_fit[fifteen_days_after_peak_f2]
    fifteen_days_magnitude_difference_f2 = np.abs(peak_magnitude_f2 - fifteen_days_magnitude_f2)

    # Magnitude difference peak - 30 days after
    thirty_days_after_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] - 30)) + amount_fit
    thirty_days_magnitude_f2 = magnitude_fit[thirty_days_after_peak_f2]
    thirty_days_magnitude_difference_f2 = np.abs(peak_magnitude_f2 - thirty_days_magnitude_f2)

    # Duration above half of the peak flux
    half_peak_flux_f2 = np.argmin(np.abs(flux_fit[f2_values_fit][(peak_fit_idx_f2 - amount_fit):] - peak_flux_fit_f2 * 0.5)) + peak_fit_idx_f2
    half_peak_time_f2 = time_fit_f2[half_peak_flux_f2]

    # Duration above a fifth of the peak flux
    fifth_peak_flux_f2 = np.argmin(np.abs(flux_fit[f2_values_fit][(peak_fit_idx_f2 - amount_fit):] - peak_flux_fit_f2 * 0.2)) + peak_fit_idx_f2
    fifth_peak_time_f2 = time_fit_f2[fifth_peak_flux_f2]

    return np.array([peak_magnitude_f1, rise_time_f1, ten_days_magnitude_difference_f1, \
                     fifteen_days_magnitude_difference_f1, thirty_days_magnitude_difference_f1, \
                     half_peak_time_f1, fifth_peak_time_f1, \
                     peak_magnitude_f2, rise_time_f2, ten_days_magnitude_difference_f2, \
                     fifteen_days_magnitude_difference_f2, thirty_days_magnitude_difference_f2, \
                     half_peak_time_f2, fifth_peak_time_f2])
    
    # return np.array([- peak_magnitude_f1, rise_time_f1, \
    #                 thirty_days_magnitude_difference_f1, \
    #                 fifth_peak_time_f1, \
    #                 - peak_magnitude_f2, rise_time_f2, \
    #                 thirty_days_magnitude_difference_f2, \
    #                 fifth_peak_time_f2])

# %%

global_parameters_OP_Ia = []

for idx in range(len(parameters_OP_Ia)):
    
    global_parameters = calculate_global_parameters(parameters_OP_Ia[idx, 0], survey, 1, parameters_OP_Ia[idx, 1:])
    global_parameters_OP_Ia.append(global_parameters)
    
global_parameters_OP_II = []

for idx in range(len(parameters_OP_II)):
    
    global_parameters = calculate_global_parameters(parameters_OP_II[idx, 0], survey, 1, parameters_OP_II[idx, 1:])
    global_parameters_OP_II.append(global_parameters)

global_parameters_TP_Ia = []

for idx in range(len(parameters_TP_Ia)):
    
    global_parameters = calculate_global_parameters(parameters_TP_Ia[idx, 0], survey, 2, parameters_TP_Ia[idx, 1:])
    global_parameters_TP_Ia.append(global_parameters)

global_parameters_TP_II = []

for idx in range(len(parameters_TP_II)):
    
    global_parameters = calculate_global_parameters(parameters_TP_II[idx, 0], survey, 2, parameters_TP_II[idx, 1:])
    global_parameters_TP_II.append(global_parameters)

global_parameters_Ia = np.concatenate((global_parameters_OP_Ia, global_parameters_TP_Ia))
global_parameters_II = np.concatenate((global_parameters_OP_II, global_parameters_TP_II))
global_parameters = np.concatenate((global_parameters_Ia, global_parameters_II))
number_of_peaks = np.concatenate((np.concatenate(([1] * len(global_parameters_OP_Ia), [2] * len(global_parameters_TP_Ia))), \
                                  np.concatenate(([1] * len(global_parameters_OP_II), [2] * len(global_parameters_TP_II)))))

global_names = ["Peak magnitude", "Rise time [days]", \
                "$\mathrm{m_{peak - 10d} - m_{peak}}$", "$\mathrm{m_{peak + 15d} - m_{peak}}$", \
                "$\mathrm{m_{peak + 30d} - m_{peak}}$", "Duration above 50 %% of peak [days]", \
                "Duration above 20 %% of peak [days]"]

# %%
# plot_correlation(global_parameters_Ia[:, 0:7], global_parameters_II[:, 0:7], survey, f1, global_names)
# plot_correlation(global_parameters_Ia[:, 7:15], global_parameters_II[:, 7:15], survey, f2, global_names)

####################################################################

# %%

def retrieve_redshift(SN_names, survey):

    redshifts = []
    file_Ia = pd.read_csv(f"Data/{survey}_Ia.csv").to_numpy()
    file_II = pd.read_csv(f"Data/{survey}_II.csv").to_numpy()
    
    for SN_id in SN_names:
    
        if SN_id in file_Ia[:, 0]:
            idx = np.where(file_Ia[:, 0] == SN_id)
            z = file_Ia[idx, 1]
            redshifts.append(float(z))

        elif SN_id in file_II[:, 0]:
            idx = np.where(file_II[:, 0] == SN_id)
            z = file_II[idx, 1]
            redshifts.append(float(z))

    return redshifts

# %%

redshifts = retrieve_redshift(parameters_one_peak[:, 0], survey)

# %%

def E(redshift):

    Omega_m = 0.3  # Matter density parameter
    Omega_Lambda = 0.7  # Dark energy density parameter

    return np.sqrt(Omega_m * (1 + redshift)**3 + Omega_Lambda)

def comoving_distance(redshift):

    c = 3.0e5  # Speed of light in km/s
    H0 = 70  # Hubble constant in km/s/Mpc

    integral, _ = quad(lambda redshift_prime: 1 / E(redshift_prime), 0, redshift)

    return (c / H0) * integral

def calculate_peak_absolute_magnitude(apparent_magnitude, redshift):

    d_C = comoving_distance(redshift)
    d_L = d_C * (1 + redshift) * 10**6

    K_correction = 2.5 * np.log10(1 + redshift)

    absolute_magnitude = apparent_magnitude - 5 * np.log10(d_L) + 5 - K_correction

    return absolute_magnitude

# %%

peak_abs_magnitude = []
for idx in range(len(redshifts)):

    peak_abs_magnitude.append(calculate_peak_absolute_magnitude(global_parameters[idx, 0], redshifts[idx]))

# %%

# PCA

def plot_PCA(parameter_values, SN_type, parameter_names):

    pca = decomposition.PCA(n_components = 2, random_state = 2804)

    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data, columns = ("Dimension 1", "Dimension 2"))   
    pca_df["SN_type"] = SN_type

    if parameter_names != None:
        
        print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))
        print("Cumulative variance explained by 2 principal components: {:.2%}".format(np.sum(pca.explained_variance_ratio_)))

        dataset_pca = pd.DataFrame(abs(pca.components_), columns = parameter_names, index = ['PC_1', 'PC_2'])
        print('\n\n', dataset_pca)

        print("\n*************** Most important features *************************")
        print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())
        print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
        print("\n******************************************************************")

    plt.figure(figsize = (8, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = pca_df[pca_df["SN_type"] == lbl]
        plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

    plt.title(f"PCA plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

def plot_PCA_with_clusters(parameter_values, SN_type, kmeans):

    pca = decomposition.PCA(n_components = 2, random_state = 2804)
    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data, columns = ("Dimension 1", "Dimension 2"))   
    pca_df["SN_type"] = SN_type

    # Create a mesh grid to cover the PCA space
    x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
    y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Predict clusters for each point in the mesh grid
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Inverse transform the PCA grid points back to the original feature space
    grid_original = pca.inverse_transform(grid_2d)

    # Now predict using KMeans on the original feature space
    grid_clusters = kmeans.predict(grid_original)
    grid_clusters = grid_clusters.reshape(xx.shape)

    # Plot the background regions (colored by KMeans predictions)
    plt.figure(figsize = (8, 6))
    plt.contourf(xx, yy, grid_clusters, levels = [-0.5, 0.5, 1.5], colors = ["tab:green", "tab:purple"], alpha = 0.3)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = pca_df[pca_df["SN_type"] == lbl]
        plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

    # Create proxy artists for the background clusters
    cluster0_patch = mpatches.Patch(color = "tab:green", label = "K-means cluster 1")
    cluster1_patch = mpatches.Patch(color = "tab:purple", label = "K-means cluster 2")

    # Labels and legend
    plt.title(f"PCA plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.3)
    plt.legend(handles = plt.legend().legend_handles  + [cluster0_patch, cluster1_patch])
    plt.show()
    
# %%

# tSNE

def plot_tSNE(parameter_values, SN_type):

    model = TSNE(n_components = 2, random_state = 2804)

    tsne_data = model.fit_transform(parameter_values)
    tsne_df = pd.DataFrame(data = tsne_data, columns = ("Dimension 1", "Dimension 2"))
    tsne_df["SN_type"] = SN_type

    plt.figure(figsize=(8, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = tsne_df[tsne_df["SN_type"] == lbl]
        plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

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

    UMAP_df = pd.DataFrame(data = UMAP_data, columns = ("Dimension 1", "Dimension 2"))
    UMAP_df["SN_type"] = SN_type

    plt.figure(figsize=(8, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = UMAP_df[UMAP_df["SN_type"] == lbl]
        plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

    plt.title(f"UMAP plot")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()
    

# %%

def number_of_clusters(parameters):

    n_clusters = np.array([2, 3, 4, 5, 10])
    silhouette_scores = []

    parameter_grid = ParameterGrid({"n_clusters": n_clusters})
    best_score = -1

    kmeans_model = KMeans()

    for p in parameter_grid:
        
        # Set number of clusters
        kmeans_model.set_params(**p)    
        kmeans_model.fit(parameters)

        ss = metrics.silhouette_score(parameters, kmeans_model.labels_)
        silhouette_scores += [ss] 
        if ss > best_score:
            best_score = ss

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align = "center", color = "#722f59", width = 0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.title("Silhouette Score", fontweight = "bold")
    plt.xlabel("Number of Clusters")
    plt.show()

    mask = np.where(silhouette_scores == best_score)
    return n_clusters[mask][0]
    
# %%

scaler = StandardScaler()

# %%

#### No redshift

parameters_one_peak_names = ["t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"]

parameters_one_peak_scaled = scaler.fit_transform(parameters_one_peak[:, 3:])

plot_PCA(parameters_one_peak_scaled, SN_labels, parameters_one_peak_names)
best_number = number_of_clusters(parameters_one_peak_scaled)

kmeans = KMeans(n_clusters = best_number)
kmeans.fit(parameters_one_peak_scaled)

plot_PCA_with_clusters(parameters_one_peak_scaled, SN_labels, kmeans)

# %%

#### With redshift

parameters_one_peak_names = ["A_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2", \
                             "M_r", "z"]

parameters_one_peak_redshift = np.concatenate((parameters_one_peak[:, 1].reshape(len(SN_labels), 1), parameters_one_peak[:, 3:], np.array(peak_abs_magnitude).reshape(len(SN_labels), 1), np.array(redshifts).reshape(len(SN_labels), 1)), axis = 1)

parameters_one_peak_scaled = scaler.fit_transform(parameters_one_peak_redshift)

plot_PCA(parameters_one_peak_scaled, SN_labels, parameters_one_peak_names)
best_number = number_of_clusters(parameters_one_peak_scaled)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(parameters_one_peak_scaled)

plot_PCA_with_clusters(parameters_one_peak_scaled, SN_labels, kmeans)

# %%

#### No redshift + double peak

parameters_two_peaks_names = ["t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"] \
                             + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]

parameters_two_peaks_scaled = scaler.fit_transform(parameters_two_peaks[:, 3:])

plot_PCA(parameters_two_peaks_scaled, SN_labels, parameters_two_peaks_names)
best_number = number_of_clusters(parameters_two_peaks_scaled)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(parameters_two_peaks_scaled)

plot_PCA_with_clusters(parameters_two_peaks_scaled, SN_labels, kmeans)

# %%

global_parameters_names = ["peak_mag_r", "rise_time_r", "mag_diff_10_r", "mag_diff_15_r", \
                           "mag_diff_30_r", "duration_50_r", "duration_20_r", \
                           "peak_mag_g", "rise_time_g", "mag_diff_10_g", "mag_diff_15_g", \
                           "mag_diff_30_g", "duration_50_g", "duration_20_g"] #, "z"]

# global_parameters_redshift = np.concatenate((global_parameters[cluster_0], np.array(redshifts).reshape(len(SN_labels[]), 1)), axis = 1)

global_parameters_scaled = scaler.fit_transform(global_parameters[cluster_0])

plot_PCA(global_parameters_scaled, SN_labels[cluster_0], global_parameters_names)
best_number = number_of_clusters(global_parameters_scaled)

kmeans = KMeans(n_clusters = best_number)
kmeans.fit(global_parameters_scaled)

plot_PCA_with_clusters(global_parameters_scaled, SN_labels[cluster_0], kmeans)

# %%

cluster_0 = np.where(kmeans.labels_ == 0)
cluster_1 = np.where(kmeans.labels_ == 1)

# %%

def plot_SN_collection(fitting_parameters, number_of_peaks):

    collection_times_f1 = []
    collection_fluxes_f1 = []

    collection_times_f2 = []
    collection_fluxes_f2 = []

    for idx in range(len(fitting_parameters)):
    
        f1 = "r"
        f2 = "g"

        time, flux, _, filters = load_ztf_data(fitting_parameters[idx, 0])

        f1_values = np.where(filters == f1)

        # Shift the light curve so that the main peak is at time = 0 MJD
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

        time -= peak_time

        amount_fit = 500
        time_fit = np.concatenate((np.linspace(-150, 500, amount_fit), np.linspace(-150, 500, amount_fit)))
        f1_values_fit = np.arange(amount_fit)
        f2_values_fit = np.arange(amount_fit) + amount_fit

        if number_of_peaks[idx] == 1:
            flux_fit = light_curve_one_peak(time_fit, fitting_parameters[idx, 1:15], peak_flux, f1_values_fit, f2_values_fit)

        elif number_of_peaks[idx] == 2:
            flux_fit = light_curve_two_peaks(time_fit, fitting_parameters[idx, 1:], peak_flux, f1_values_fit, f2_values_fit)

        time_fit += peak_time

        peak_main_idx_f1 = np.argmax(flux_fit[f1_values_fit])
        time_fit_f1 = time_fit[f1_values_fit] - time_fit[peak_main_idx_f1]
        collection_times_f1.append(np.array(time_fit_f1))

        peak_main_idx_f2 = np.argmax(flux_fit[f1_values_fit])
        time_fit_f2 = time_fit[f2_values_fit] - time_fit[peak_main_idx_f2]
        collection_times_f2.append(np.array(time_fit_f2))

        # Reshape the data so that the flux is between 0 and 1 micro Jy
        # flux_fit_min_f1 = np.copy(np.min(flux_fit[f1_values_fit]))
        flux_fit_max_f1 = np.copy(np.max(flux_fit[f1_values_fit]))
        flux_fit_f1 = (flux_fit[f1_values_fit]) / flux_fit_max_f1
        collection_fluxes_f1.append(np.array(flux_fit_f1))

        # flux_fit_min_f2 = np.copy(np.min(flux_fit[f2_values_fit]))
        flux_fit_max_f2 = np.copy(np.max(flux_fit[f2_values_fit]))
        flux_fit_f2 = (flux_fit[f2_values_fit]) / flux_fit_max_f2
        collection_fluxes_f2.append(np.array(flux_fit_f2))

    return collection_times_f1, collection_times_f2, collection_fluxes_f1, collection_fluxes_f2

# %%

collection_times_f1, collection_times_f2, collection_fluxes_f1, collection_fluxes_f2 = plot_SN_collection(parameters_two_peaks, number_of_peaks)

# %%

plt.scatter(np.array(collection_times_f1)[cluster_0], np.array(collection_fluxes_f1)[cluster_0], s = 1)

# plt.scatter(np.array(collection_times_f1)[cluster_1], np.array(collection_fluxes_f1)[cluster_1], s = 1)
plt.show()

# %%

plt.scatter(np.array(collection_times_f1)[len(parameters_two_peaks_Ia):], np.array(collection_fluxes_f1)[len(parameters_two_peaks_Ia):], s = 1)
# plt.scatter(np.array(collection_times_f1)[:len(parameters_two_peaks_Ia)], np.array(collection_fluxes_f1)[:len(parameters_two_peaks_Ia)], s = 1)
# plt.scatter(np.array(collection_times_f1)[cluster_1], np.array(collection_fluxes_f1)[cluster_1], s = 0.5)
# plt.scatter(np.array(collection_times_f1)[10], np.array(collection_fluxes_f1)[10], s = 1)

plt.show()

# %%
parameters_one_peak[cluster_1, 0]
# %%
# ["ZTF19aceqlxc", "ZTF19acykaae", "ZTF18aamftst"] in S23 as possible Ia-CSM and also in cluster 0
# "ZTF19acvkibv" in S23 as possible Ia-CSM but not in cluster 0 0

# ["2019kep"] in S23 as possible Ia-CSM 
# Add well known SNe

# mag_diff_15_r mag_diff_30_r duration_50_r duration_20_r mag_diff_15_g mag_diff_30_g duration_20_g
# The shape of the light curve afterwards is important
# Maybe it removes plateau LC