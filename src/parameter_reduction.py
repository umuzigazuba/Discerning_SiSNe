# %%

from src.data_processing import ztf_load_data, atlas_load_data, atlas_micro_flux_to_magnitude
from src.parameter_estimation import light_curve_one_peak, light_curve_two_peaks

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
import seaborn as sn
import pandas as pd
import os
import csv

plt.rcParams["text.usetex"] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

# Retrieve parameters 

def retrieve_parameters(SN_names, survey):

    parameters_OP = []
    parameters_TP = []

    for SN_id in SN_names:

        if os.path.isfile(f"Data/Analytical_parameters/{survey}/two_peaks/{SN_id}_parameters_TP.npy"):
            data = np.load(f"Data/Analytical_parameters/{survey}/two_peaks/{SN_id}_parameters_TP.npy")
            parameters_TP.append([SN_id, *data])

        else:
            data = np.load(f"Data/Analytical_parameters/{survey}/one_peak/{SN_id}_parameters_OP.npy")
            parameters_OP.append([SN_id, *data])

    parameters_OP = np.array(parameters_OP, dtype = object)
    parameters_TP = np.array(parameters_TP, dtype = object)

    return parameters_OP, parameters_TP

def retrieve_parameters_one_peak(SN_names, survey):

    parameters_OP = []

    for SN_id in SN_names:

        if os.path.isfile(f"Data/Analytical_parameters/{survey}/one_peak/{SN_id}_parameters_OP.npy"):
            data = np.load(f"Data/Analytical_parameters/{survey}/one_peak/{SN_id}_parameters_OP.npy")
            parameters_OP.append([SN_id, *data])

    parameters_OP = np.array(parameters_OP, dtype = object)

    return parameters_OP

# %%

# Retrieve reduced chi squared values

def retrieve_red_chi_squared(file_name, SN_names_I, SN_names_II):

    red_chi_squared_values_Ia = []
    red_chi_squared_values_II = []

    with open(file_name, "r") as file:

        reader = csv.reader(file)

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
        time, flux, fluxerr, filters = ztf_load_data(SN_id)

    elif survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, fluxerr, filters = atlas_load_data(SN_id)

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
    bins = np.logspace(np.log10(min_bin), np.log10(max_bin), 30)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = colours["orange"], histtype = "bar", alpha = 0.4, zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = colours["blue"], histtype = "bar", alpha = 0.4, zorder = 5)

    plt.hist(red_chi_squared_Ia, bins = bins, linewidth = 2, color = colours["orange"], histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
    plt.hist(red_chi_squared_II, bins = bins, linewidth = 2, color = colours["blue"], histtype = "step",  fill = False, label = "SNe IIn", zorder = 5)

    plt.axvline(x = percentile_95, color = "black", linestyle = "dashed", label = f"95th percentile = {percentile_95:.2f}")

    plt.xlabel(r"$\mathrm{X}^{2}_{red}$", fontsize = 13)
    plt.ylabel("N", fontsize = 13)
    plt.title(r"$\mathrm{X}^{2}_{red}$" + f" distribution of {survey} SNe.")
    plt.xscale("log")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.savefig(f"Plots/Results/red_chi_squared_{survey}.png", dpi = 300, bbox_inches = "tight")
    plt.show()

# %%

def plot_correlation(parameter_values_Ia, parameter_values_II, survey, filter, parameters):

    for idx_1 in range(len(parameters) - 1):
        for idx_2 in range(idx_1 + 1, len(parameters)): 

            plt.scatter(parameter_values_Ia[:, idx_1], parameter_values_Ia[:, idx_2], c = colours["orange"], label = "SNe Ia-CSM", zorder = 10)
            plt.scatter(parameter_values_II[:, idx_1], parameter_values_II[:, idx_2], c = colours["blue"], label = "SNe IIn", zorder = 5)
            
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
            sn.kdeplot(x = parameter_values_Ia[:, idx_1], y = parameter_values_Ia[:, idx_2], thresh = 0.3, color = colours["orange"], label = "SNe Ia-CSM", zorder = 10)
            sn.kdeplot(x = parameter_values_II[:, idx_1], y = parameter_values_II[:, idx_2], thresh = 0.3, color = colours["blue"], label = "SNe IIn", zorder = 5)

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

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, linewidth = 2, color = colours["orange"], histtype = "bar", alpha = 0.4, zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, linewidth = 2, color = colours["blue"], histtype = "bar", alpha = 0.4, zorder = 5)

        plt.hist(parameter_values_Ia[:, idx_1], bins = bins, linewidth = 2, color = colours["orange"], histtype = "step",  fill = False, label = "SNe Ia-CSM", zorder = 10)
        plt.hist(parameter_values_II[:, idx_1], bins = bins, linewidth = 2, color = colours["blue"], histtype = "step", fill = False, label = "SNe IIn", zorder = 5)

        plt.xlabel(parameters[idx_1], fontsize = 13)
        plt.ylabel("N", fontsize = 13)
        plt.title(f"Parameter distribution of {survey} SNe in the {filter}-filter.")
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

# %%

# plot_correlation_contour(parameters_one_peak_Ia[:, 1:7], parameters_one_peak_II[:, 1:7], survey, f1, parameters)
# plot_correlation_contour(parameters_one_peak_Ia[:, 8:14], parameters_one_peak_II[:, 8:14], survey, f2, parameters)


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

        time, flux, _, filters = ztf_load_data(SN_id)

    if survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, _, filters = atlas_load_data(SN_id)

    f1_values = np.where(filters == f1)

    # Shift the light curve so that the main peak is at time = 0 MJD
    peak_main_idx = np.argmax(flux[f1_values])
    peak_time = np.copy(time[peak_main_idx])
    peak_flux = np.copy(flux[peak_main_idx])

    time -= peak_time

    amount_fit = 1000
    time_fit = np.concatenate((np.linspace(time.min() - 75, time.max() + 75, amount_fit), np.linspace(time.min() - 75, time.max() + 75, amount_fit)))
    f1_values_fit = np.arange(amount_fit)
    f2_values_fit = np.arange(amount_fit) + amount_fit

    if peak_number == 1:
        flux_fit = light_curve_one_peak(time_fit, parameter_values[:14], peak_flux, f1_values_fit, f2_values_fit)

    elif peak_number == 2:
        flux_fit = light_curve_two_peaks(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

    magnitude_fit, _ = atlas_micro_flux_to_magnitude(flux_fit, flux_fit)

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

####################################################################

# %%

def retrieve_redshift(SN_names, survey):

    survey_information = pd.read_csv(f"Data/{survey}_info.csv")
    redshifts = []

    for name in SN_names:
        
        SN_idx = np.where(survey_information["Disc. Internal Name"] == name)
        
        z = survey_information["Redshift"].values[SN_idx][0]
        redshifts.append(float(z))

    return redshifts

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

# %%

if __name__ == '__main__':
    
    survey = "ZTF"

    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

    elif survey == "ATLAS":
        f1 = "o"
        f2 = 'c'

    # Load the SN names
    SN_names_Ia = np.loadtxt(f"Data/{survey}_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str")
    SN_names_II = np.loadtxt(f"Data/{survey}_SNe_IIn.txt", delimiter = ",", dtype = "str")
    print(len(SN_names_Ia) + len(SN_names_II))
    # %%

    # Retrieve the reduced chi squared values
    red_chi_squared_values_OP_Ia, red_chi_squared_values_OP_II = retrieve_red_chi_squared(f"Data/Analytical_parameters/{survey}/one_peak/red_chi_squared_OP.csv", SN_names_Ia, SN_names_II)
    red_chi_squared_values_TP_Ia, red_chi_squared_values_TP_II = retrieve_red_chi_squared(f"Data/Analytical_parameters/{survey}/two_peaks/red_chi_squared_TP.csv", SN_names_Ia, SN_names_II)

    if len(red_chi_squared_values_TP_Ia) != 0:
        red_chi_squared_values_Ia = np.concatenate((red_chi_squared_values_OP_Ia, red_chi_squared_values_TP_Ia), axis = 0)
    else:
        red_chi_squared_values_Ia = red_chi_squared_values_OP_Ia
    red_chi_squared_values_II = np.concatenate((red_chi_squared_values_OP_II, red_chi_squared_values_TP_II), axis = 0)
    red_chi_squared_values = np.concatenate((red_chi_squared_values_Ia, red_chi_squared_values_II))
    
    # %%
    percentile_95 = np.percentile(red_chi_squared_values[:, 1], 95)
    print(percentile_95)

    # Remove light curves with reduced chi squared larger than the 95th percentile
    cut_light_curves = np.where(red_chi_squared_values[:, 1] > percentile_95)
    print(len(cut_light_curves[0]), len(red_chi_squared_values[:, 1]))

    cut_light_curves_Ia = np.where(np.isin(SN_names_Ia, red_chi_squared_values[cut_light_curves, 0][0]))[0]
    cut_light_curves_II = np.where(np.isin(SN_names_II, red_chi_squared_values[cut_light_curves, 0][0]))[0]

    print(len(SN_names_Ia) + len(SN_names_II))
    SN_names_Ia = np.delete(SN_names_Ia, cut_light_curves_Ia)
    SN_names_II = np.delete(SN_names_II, cut_light_curves_II)
    print(len(SN_names_Ia) + len(SN_names_II))

    SN_labels = np.array(["SN Ia CSM"] * len(SN_names_Ia) + ["SN IIn"] * len(SN_names_II))
    SN_labels_color = np.array([0] * len(SN_names_Ia) + [1] * len(SN_names_II))

    plot_red_chi_squared(red_chi_squared_values_Ia[:, 1], red_chi_squared_values_II[:, 1], percentile_95, survey)

    # %%

    # Retrieve the parameters
    fitting_parameters_OP_Ia, fitting_parameters_TP_Ia = retrieve_parameters(SN_names_Ia, survey)
    fitting_parameters_OP_II, fitting_parameters_TP_II = retrieve_parameters(SN_names_II, survey)

    if len(fitting_parameters_TP_Ia) != 0:
        fitting_parameters_Ia = np.concatenate((np.concatenate((fitting_parameters_OP_Ia, np.zeros((len(fitting_parameters_OP_Ia), 6))), axis = 1), fitting_parameters_TP_Ia))
    
    else:
        fitting_parameters_Ia = np.concatenate((fitting_parameters_OP_Ia, np.zeros((len(fitting_parameters_OP_Ia), 6))), axis = 1)
    
    fitting_parameters_II = np.concatenate((np.concatenate((fitting_parameters_OP_II, np.zeros((len(fitting_parameters_OP_II), 6))), axis = 1), fitting_parameters_TP_II))
    fitting_parameters = np.concatenate((fitting_parameters_Ia, fitting_parameters_II))
    
    fitting_parameters_one_peak = retrieve_parameters_one_peak(fitting_parameters[:, 0], survey)

    number_of_peaks = np.concatenate((np.concatenate(([1] * len(fitting_parameters_OP_Ia), [2] * len(fitting_parameters_TP_Ia))), \
                                    np.concatenate(([1] * len(fitting_parameters_OP_II), [2] * len(fitting_parameters_TP_II)))))
    
    fitting_parameter_names = ["$\mathrm{A}$", "$\mathrm{t_{0}}$", "$\mathrm{t_{rise}}$", "$\mathrm{\gamma}$", r"$\mathrm{\beta}$", "$\mathrm{t_{fall}}$"]

    # plot_correlation(fitting_parameters_Ia[:, 1:7], fitting_parameters_II[:, 1:7], survey, f1, fitting_parameter_names)
    # plot_correlation(fitting_parameters_Ia[:, 8:14], fitting_parameters_II[:, 8:14], survey, f2, fitting_parameter_names)

    # plot_distribution(fitting_parameters_Ia[:, 1:7], fitting_parameters_II[:, 1:7], survey, f1, fitting_parameter_names)
    # plot_distribution(fitting_parameters_Ia[:, 8:14], fitting_parameters_II[:, 8:14], survey, f2, fitting_parameter_names)

    # %%

    global_parameters_Ia = []
    global_parameters_II = []

    for idx in range(len(fitting_parameters_Ia)):
        

        try:
            parameter_values = calculate_global_parameters(fitting_parameters_Ia[idx, 0], survey, number_of_peaks[idx], fitting_parameters_Ia[idx, 1:])
            global_parameters_Ia.append(parameter_values)

        except ValueError: 
            # Remove light curve from list because it cannot be fit
            file_name = f"Data/{survey}_SNe_Ia_CSM.txt"
                
            with open(file_name, "r") as file:             
                SN_names = file.readlines()

            with open(file_name, "w") as file:
                for name in SN_names:

                    if name.strip("\n") != fitting_parameters_Ia[idx, 0]:
                        file.write(name)

    global_parameters_Ia = np.array(global_parameters_Ia)

    for idx in range(len(fitting_parameters_II)):
    
        try:
            parameter_values = calculate_global_parameters(fitting_parameters_II[idx, 0], survey, number_of_peaks[len(fitting_parameters_Ia) + idx], fitting_parameters_II[idx, 1:])
            global_parameters_II.append(parameter_values)

        except:
            # Remove light curve from list because it cannot be fit
            file_name = f"Data/{survey}_SNe_IIn.txt"
                
            with open(file_name, "r") as file:             
                SN_names = file.readlines()

            with open(file_name, "w") as file:
                for name in SN_names:

                    if name.strip("\n") != fitting_parameters_II[idx, 0]:
                        file.write(name)

    global_parameters_II = np.array(global_parameters_II)

    global_parameters = np.concatenate((global_parameters_Ia, global_parameters_II))

    # %%

    number_of_peaks_one_peak = [1] * len(fitting_parameters_one_peak)
    global_parameters_one_peak = np.array([calculate_global_parameters(fitting_parameters_one_peak[idx, 0], survey, number_of_peaks_one_peak[idx], fitting_parameters_one_peak[idx, 1:]) for idx in range(len(fitting_parameters_one_peak[:, 0]))])

    global_names = ["Peak magnitude", "Rise time [days]", "$\mathrm{m_{peak - 10d} - m_{peak}}$", "$\mathrm{m_{peak + 15d} - m_{peak}}$", \
                    "$\mathrm{m_{peak + 30d} - m_{peak}}$", "Duration above 50 \% of peak [days]", "Duration above 20 \% of peak [days]"]
    
    # redshifts = retrieve_redshift(fitting_parameters[:, 0], survey)
    
    # peak_abs_magnitude = []
    # for idx in range(len(redshifts)):

    #     peak_abs_magnitude.append(calculate_peak_absolute_magnitude(global_parameters[idx, 0], redshifts[idx]))

    # plot_correlation(global_parameters_Ia[:, 0:7], global_parameters_II[:, 0:7], survey, f1, global_names)
    # plot_correlation(global_parameters_Ia[:, 7:15], global_parameters_II[:, 7:15], survey, f2, global_names)
    
    # plot_distribution(global_parameters_Ia[:, 0:7], global_parameters_II[:, 0:7], survey, f1, global_names)
    # plot_distribution(global_parameters_Ia[:, 7:15], global_parameters_II[:, 7:15], survey, f2, global_names)

    # %%

    np.save(f"Data/Input_ML/{survey}/fitting_parameters.npy", fitting_parameters)
    np.save(f"Data/Input_ML/{survey}/fitting_parameters_one_peak.npy", fitting_parameters_one_peak)
    np.save(f"Data/Input_ML/{survey}/global_parameters.npy", global_parameters)
    np.save(f"Data/Input_ML/{survey}/global_parameters_one_peak.npy", global_parameters_one_peak)
    np.save(f"Data/Input_ML/{survey}/number_of_peaks.npy", number_of_peaks)
    np.save(f"Data/Input_ML/{survey}/SN_labels.npy", SN_labels)
    np.save(f"Data/Input_ML/{survey}/SN_labels_color.npy", SN_labels_color)

# %%
