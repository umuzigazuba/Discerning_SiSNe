# %%
from data_processing import ztf_load_data, atlas_load_data, data_augmentation
import pymultinest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv

from scipy.stats import truncnorm
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.optimize import curve_fit

plt.rcParams["text.usetex"] = True

# %%

# Load light curve data points 
ztf_names_sn_Ia_CSM = np.loadtxt("Data/ZTF_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str")
ztf_names_sn_IIn = np.loadtxt("Data/ZTF_SNe_IIn.txt", delimiter = ",", dtype = "str")
ztf_names = np.concatenate((ztf_names_sn_Ia_CSM, ztf_names_sn_IIn))

atlas_names_sn_Ia_CSM = np.loadtxt("Data/ATLAS_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str")
atlas_names_sn_IIn = np.loadtxt("Data/ATLAS_SNe_IIn.txt", delimiter = ",", dtype = "str")
atlas_names = np.concatenate((atlas_names_sn_Ia_CSM, atlas_names_sn_IIn))

# %%

global mean_one_peak, std_one_peak, mean_two_peaks, std_two_peaks, parameter_bounds_one_peak, parameter_bounds_gaussian, parameter_bounds_two_peaks
global f1, f2
global time, flux, fluxerr, filters, f1_values, f2_values, peak_main_idx, f1_values_peak, f2_values_peak, main_flux_second_peak

# Parameters of a light curve with one peak
parameters_one_peak = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                       "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"]

parameters_two_peaks = parameters_one_peak + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]

n_parameters_one_peak = len(parameters_one_peak) 
n_parameters_two_peaks = len(parameters_two_peaks) 

# %%

# Initial guess of the values of the parameters
# From the Superphot+  paper
initial_mean_one_peak = [0.0957, -17.878, 0.6664, 1.4258, 0.00833, 1.5261, -1.6629,  -0.0766, -3.4065, -0.1510, -0.0452, -0.2068, -0.1486, -0.1509] 
initial_std_one_peak = [0.0575, 9.916, 0.4250, 0.5079, 0.00585, 0.5057, 0.5578, 0.1096, 4.566, 0.1927, 0.1705, 0.2704, 0.2642, 0.2542] 

mean_one_peak = initial_mean_one_peak
std_one_peak = initial_std_one_peak

initial_mean_two_peaks = initial_mean_one_peak + [0.63763033, 118.0002509, 51.6945267, 0.88386988, 0.96791228, 0.89671384]
initial_std_two_peaks = initial_std_one_peak + [0.27643953, 92.80604182, 51.55866317, 0.20874114, 0.18211842, 0.38862772]

mean_two_peaks = initial_mean_two_peaks
std_two_peaks = initial_std_two_peaks

parameter_bounds_one_peak = ([-0.3, -100, -2, 0, 0, 0, -3, -1, -50, -1.5, -1.5, -2, -1.5, -1.5], [0.5, 30, 4, 3.5, 0.03, 4, -0.8, 1, 30, 1.5, 1.5, 1, 1.5, -1])

parameter_bounds_gaussian = ([0, -300, 0, -0.5, -0.5, -1.5], [1.5, 300, 250, 1.5, 5.0, 5.0])

parameter_bounds_two_peaks = ([-0.3, -100, -2, 0, 0, 0, -3, -1, -50, -1.5, -1.5, -2, -1.5, -1.5, 0, -300, 0, -0.5, -0.5, -1.5], [0.5, 30, 4, 3.5, 0.03, 4, -0.8, 1, 30, 1.5, 1.5, 1, 1.5, -1, 1.5, 300, 250, 1.5, 5.0, 5.0])

def prior_one_peak(cube, ndim, nparams):

    ### First peak

    # A 1
    min = (-0.3 - mean_one_peak[0]) / std_one_peak[0]
    max = (0.5 - mean_one_peak[0]) / std_one_peak[0]
    cube[0] = truncnorm.ppf(cube[0], min, max, mean_one_peak[0], std_one_peak[0])

    # t_0 1
    min = (-100.0 - mean_one_peak[1]) / std_one_peak[1]
    max = (30.0 - mean_one_peak[1]) / std_one_peak[1]
    cube[1] = truncnorm.ppf(cube[1], min, max, mean_one_peak[1], std_one_peak[1])

    # t_rise 1
    min = (-2.0 - mean_one_peak[2]) / std_one_peak[2]
    max = (4.0 - mean_one_peak[2]) / std_one_peak[2]
    cube[2] = truncnorm.ppf(cube[2], min, max, mean_one_peak[2], std_one_peak[2])

    # gamma 1
    min = (0.0 - mean_one_peak[3]) / std_one_peak[3]
    max = (3.5 - mean_one_peak[3]) / std_one_peak[3]
    cube[3] = truncnorm.ppf(cube[3], min, max, mean_one_peak[3], std_one_peak[3])

    # beta 1
    min = (0.0 - mean_one_peak[4]) / std_one_peak[4]
    max = (0.03 - mean_one_peak[4]) / std_one_peak[4]
    cube[4] = truncnorm.ppf(cube[4], min, max, mean_one_peak[4], std_one_peak[4])

    # t_fall 1
    min = (0.0 - mean_one_peak[5]) / std_one_peak[5]
    max = (4.0 - mean_one_peak[5]) / std_one_peak[5]
    cube[5] = truncnorm.ppf(cube[5], min, max, mean_one_peak[5], std_one_peak[5])

    # error 1
    min = (-3.0 - mean_one_peak[6]) / std_one_peak[6]
    max = (-0.8 - mean_one_peak[6]) / std_one_peak[6]
    cube[6] = truncnorm.ppf(cube[6], min, max, mean_one_peak[6], std_one_peak[6])

    # A 2
    min = (-1.0 - mean_one_peak[7]) / std_one_peak[7]
    max = (1.0 - mean_one_peak[7]) / std_one_peak[7]
    cube[7] = truncnorm.ppf(cube[7], min, max, mean_one_peak[7], std_one_peak[7])

    # t_0 2
    min = (-50.0 - mean_one_peak[8]) /std_one_peak[8] 
    max = (30.0 - mean_one_peak[8]) / std_one_peak[8]
    cube[8] = truncnorm.ppf(cube[8], min, max, mean_one_peak[8], std_one_peak[8])

    # t_rise 2
    min = (-1.5 - mean_one_peak[9]) / std_one_peak[9]
    max = (1.5 - mean_one_peak[9]) / std_one_peak[9]
    cube[9] = truncnorm.ppf(cube[9], min, max, mean_one_peak[9], std_one_peak[9])

    # gamma 2
    min = (-1.5 - mean_one_peak[10]) / std_one_peak[10]
    max = (1.5 - mean_one_peak[10]) / std_one_peak[10]
    cube[10] = truncnorm.ppf(cube[10], min, max, mean_one_peak[10], std_one_peak[10])

    # beta 2
    min = (-2.0 - mean_one_peak[11]) / std_one_peak[11]
    max = (1.0 - mean_one_peak[11]) / std_one_peak[11]
    cube[11] = truncnorm.ppf(cube[11], min, max, mean_one_peak[11], std_one_peak[11])

    # t_fall 2
    min = (-1.5 - mean_one_peak[12]) / std_one_peak[12]
    max = (1.5 - mean_one_peak[12]) / std_one_peak[12]
    cube[12] = truncnorm.ppf(cube[12], min, max, mean_one_peak[12], std_one_peak[12])

    # error 2
    min = (-1.5 - mean_one_peak[13]) / std_one_peak[13]
    max = (-1.0 - mean_one_peak[13]) / std_one_peak[13]
    cube[13] = truncnorm.ppf(cube[13], min, max, mean_one_peak[13], std_one_peak[13])

    return cube

def prior_two_peaks(cube, ndim, nparams):

    ### First peak

    # A 1
    min = (-0.3 - mean_two_peaks[0]) / std_two_peaks[0]
    max = (0.5 - mean_two_peaks[0]) / std_two_peaks[0]
    cube[0] = truncnorm.ppf(cube[0], min, max, mean_two_peaks[0], std_two_peaks[0])

    # t_0 1
    min = (-100.0 - mean_two_peaks[1]) / std_two_peaks[1]
    max = (30.0 - mean_two_peaks[1]) / std_two_peaks[1]
    cube[1] = truncnorm.ppf(cube[1], min, max, mean_two_peaks[1], std_two_peaks[1])

    # t_rise 1
    min = (-2.0 - mean_two_peaks[2]) / std_two_peaks[2]
    max = (4.0 - mean_two_peaks[2]) / std_two_peaks[2]
    cube[2] = truncnorm.ppf(cube[2], min, max, mean_two_peaks[2], std_two_peaks[2])

    # gamma 1
    min = (0.0 - mean_two_peaks[3]) / std_two_peaks[3]
    max = (3.5 - mean_two_peaks[3]) / std_two_peaks[3]
    cube[3] = truncnorm.ppf(cube[3], min, max, mean_two_peaks[3], std_two_peaks[3])

    # beta 1
    min = (0.0 - mean_two_peaks[4]) / std_two_peaks[4]
    max = (0.03 - mean_two_peaks[4]) / std_two_peaks[4]
    cube[4] = truncnorm.ppf(cube[4], min, max, mean_two_peaks[4], std_two_peaks[4])

    # t_fall 1
    min = (0.0 - mean_two_peaks[5]) / std_two_peaks[5]
    max = (4.0 - mean_two_peaks[5]) / std_two_peaks[5]
    cube[5] = truncnorm.ppf(cube[5], min, max, mean_two_peaks[5], std_two_peaks[5])

    # error 1
    min = (-3.0 - mean_two_peaks[6]) / std_two_peaks[6]
    max = (-0.8 - mean_two_peaks[6]) / std_two_peaks[6]
    cube[6] = truncnorm.ppf(cube[6], min, max, mean_two_peaks[6], std_two_peaks[6])

    # A 2
    min = (-1.0 - mean_two_peaks[7]) / std_two_peaks[7]
    max = (1.0 - mean_two_peaks[7]) / std_two_peaks[7]
    cube[7] = truncnorm.ppf(cube[7], min, max, mean_two_peaks[7], std_two_peaks[7])

    # t_0 2
    min = (-50.0 - mean_two_peaks[8]) /std_two_peaks[8] 
    max = (30.0 - mean_two_peaks[8]) / std_two_peaks[8]
    cube[8] = truncnorm.ppf(cube[8], min, max, mean_two_peaks[8], std_two_peaks[8])

    # t_rise 2
    min = (-1.5 - mean_two_peaks[9]) / std_two_peaks[9]
    max = (1.5 - mean_two_peaks[9]) / std_two_peaks[9]
    cube[9] = truncnorm.ppf(cube[9], min, max, mean_two_peaks[9], std_two_peaks[9])

    # gamma 2
    min = (-1.5 - mean_two_peaks[10]) / std_two_peaks[10]
    max = (1.5 - mean_two_peaks[10]) / std_two_peaks[10]
    cube[10] = truncnorm.ppf(cube[10], min, max, mean_two_peaks[10], std_two_peaks[10])

    # beta 2
    min = (-2.0 - mean_two_peaks[11]) / std_two_peaks[11]
    max = (1.0 - mean_two_peaks[11]) / std_two_peaks[11]
    cube[11] = truncnorm.ppf(cube[11], min, max, mean_two_peaks[11], std_two_peaks[11])

    # t_fall 2
    min = (-1.5 - mean_two_peaks[12]) / std_two_peaks[12]
    max = (1.5 - mean_two_peaks[12]) / std_two_peaks[12]
    cube[12] = truncnorm.ppf(cube[12], min, max, mean_two_peaks[12], std_two_peaks[12])

    # error 2
    min = (-1.5 - mean_two_peaks[13]) / std_two_peaks[13]
    max = (-1.0 - mean_two_peaks[13]) / std_two_peaks[13]
    cube[13] = truncnorm.ppf(cube[13], min, max, mean_two_peaks[13], std_two_peaks[13])

    ### Second peak 

    # amp 1
    min = (0.0 - mean_two_peaks[14]) /std_two_peaks[14] 
    max = (1.5 - mean_two_peaks[14]) / std_two_peaks[14]
    cube[14] = truncnorm.ppf(cube[14], min, max, mean_two_peaks[14], std_two_peaks[14])

    # mu 1
    min = (0.0 - mean_two_peaks[15]) / std_two_peaks[15]
    max = (300.0 - mean_two_peaks[15]) / std_two_peaks[15]
    cube[15] = truncnorm.ppf(cube[15], min, max, mean_two_peaks[15], std_two_peaks[15])

    # std_two_peaks 1
    min = (0.0 - mean_two_peaks[16]) / std_two_peaks[16]
    max = (250.0 - mean_two_peaks[16]) / std_two_peaks[16]
    cube[16] = truncnorm.ppf(cube[16], min, max, mean_two_peaks[16], std_two_peaks[16])

    # amp 2
    min = (0 - mean_two_peaks[17]) / std_two_peaks[17]
    max = (1.5 - mean_two_peaks[17]) / std_two_peaks[17]
    cube[17] = truncnorm.ppf(cube[17], min, max, mean_two_peaks[17], std_two_peaks[17])

    # mu 2
    min = (0 - mean_two_peaks[18]) / std_two_peaks[18]
    max = (1.5 - mean_two_peaks[18]) / std_two_peaks[18]
    cube[18] = truncnorm.ppf(cube[18], min, max, mean_two_peaks[18], std_two_peaks[18])

    # std_two_peaks 2
    min = (-1.5 - mean_two_peaks[19]) / std_two_peaks[19]
    max = (2.0 - mean_two_peaks[19]) / std_two_peaks[19]
    cube[19] = truncnorm.ppf(cube[19], min, max, mean_two_peaks[19], std_two_peaks[19])

    return cube
# %%

########################################## PRE-PROCESSING ####################################################

### Data augmentation ###

def augment_data(survey, peak_time, amount_aug):

    global time, flux, fluxerr, filters, f1_values, f2_values

    # Augment the amount of data points using Gaussian Processing (fulu)
    _, _, augmentation = data_augmentation(survey, time, flux, fluxerr, filters, "GP")

    time_aug, flux_aug, fluxerr_aug, filters_aug = augmentation.augmentation(time.min(), time.max(), n_obs = amount_aug)

    f1_values_aug = np.where(filters_aug == f1)
    f2_values_aug = np.where(filters_aug == f2)

    # Remove the augmented data after the last datapoint in each filter
    if f1 in filters:
        out_of_range_f1 = np.where((time_aug[f1_values_aug] + peak_time > time[f1_values].max() + peak_time))[0]
    else:
        out_of_range_f1 = np.array([])

    if f2 in filters:
        out_of_range_f2 = np.where((time_aug[f2_values_aug] + peak_time > time[f2_values].max() + peak_time))[0] + amount_aug
    else:
        out_of_range_f2 = np.array([])

    if len(np.concatenate((out_of_range_f1, out_of_range_f2))) != 0 :
        time_aug = np.delete(time_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        flux_aug = np.delete(flux_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        fluxerr_aug = np.delete(fluxerr_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        filters_aug = np.delete(filters_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))

    # Fulu will create data points if a filter does not contain any data
    # Remove these data points
    if f1 not in filters:
        time_aug = np.delete(time_aug, np.arange(0, amount_aug))
        flux_aug = np.delete(flux_aug, np.arange(0, amount_aug))
        fluxerr_aug = np.delete(fluxerr_aug, np.arange(0, amount_aug))
        filters_aug = np.delete(filters_aug, np.arange(0, amount_aug))
    
    if f2 not in filters:
        time_aug = np.delete(time_aug, np.arange(0, amount_aug) + amount_aug)
        flux_aug = np.delete(flux_aug, np.arange(0, amount_aug) + amount_aug)
        fluxerr_aug = np.delete(fluxerr_aug, np.arange(0, amount_aug) + amount_aug)
        filters_aug = np.delete(filters_aug, np.arange(0, amount_aug) + amount_aug)

    return time_aug, flux_aug, fluxerr_aug, filters_aug

### Find peaks ###

def find_main_and_second_peak(time, time_aug, flux_aug, peaks_max):

    # Calculate prominences
    prominences = peak_prominences(flux_aug, peaks_max)[0]

    # Calculate peak widths in data points (full width at half maximum)
    widths = peak_widths(flux_aug, peaks_max, rel_height = 0.5)[0]
    number_data_points = [len(np.where(np.abs(time - time_aug[peaks_max][idx]) < widths[idx])[0]) \
                        for idx in range(len(peaks_max))]

    # Extract peak heights
    peak_heights = flux_aug[peaks_max]

    # Create a list of tuples containing all information for each peak
    peak_info = list(zip(peaks_max, peak_heights, prominences, number_data_points))

    # Sort peaks by height (flux value) to find the second largest
    sorted_by_height = sorted(peak_info, key=lambda x: x[1], reverse=True)

    # Extract the largest peak
    largest_peak = sorted_by_height[0]

    # Filter peaks to find the one with the second highest prominence, most data points, and highest flux value
    second_largest_peak_candidates = [peak for peak in peak_info if peak != largest_peak]

    # Sort by prominence, number of data points (widths), and height
    sorted_candidates = sorted(second_largest_peak_candidates, key = lambda x: (x[2], x[3], x[1]), reverse = True)

    # Select the best candidate
    best_candidate = sorted_candidates[-1]

    return largest_peak[0], best_candidate[0]

def find_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug):

    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    # Reshape the data so that the flux is between 0 and 1 micro Jy
    flux_min = np.copy(np.min(flux_aug))
    flux_max = np.copy(np.max(flux_aug))

    flux = (flux - flux_min) / flux_max
    fluxerr = (fluxerr) / flux_max
    flux_aug = (flux_aug - flux_min) / flux_max
    fluxerr_aug = (fluxerr_aug) / flux_max

    # Filter 1
    # Find the extrema of the augmented data
    peaks_max_f1, _ = find_peaks(flux_aug[f1_values_aug], prominence = 0.05)
    peaks_min_f1, _ = find_peaks(-flux_aug[f1_values_aug], prominence = 0.05)

    if len(peaks_max_f1) > 1:
        # If more than one maxima, identify the main and second peaks
        main_peak_f1, second_peak_f1 = find_main_and_second_peak(time[f1_values], time_aug[f1_values_aug], flux_aug[f1_values_aug], peaks_max_f1)    

        # Find the minimum closes to the second peak 
        # This is equal to the base of the second peak
        min_distance_to_second_peak_f1 = np.abs(time_aug[f1_values_aug][peaks_min_f1] - time_aug[f1_values_aug][second_peak_f1])
        min_second_f1 = peaks_min_f1[np.argmin(min_distance_to_second_peak_f1)]

        # Calculate the widths of the second peak
        peak_width_f1 = np.abs(time_aug[f1_values_aug][second_peak_f1] - time_aug[f1_values_aug][min_second_f1])

    elif len(peaks_max_f1) == 1 and len(peaks_min_f1) == 1:
        # If one peak is hidden, consider the visible peak to be the second one
        main_peak_f1 = -1
        second_peak_f1 = peaks_max_f1[0]
        min_second_f1 = peaks_min_f1[0]
        peak_width_f1 = np.abs(time_aug[f1_values_aug][second_peak_f1] - time_aug[f1_values_aug][min_second_f1])

    elif len(peaks_max_f1) == 1:
        # If only one maxima
        main_peak_f1 = peaks_max_f1[0]
        second_peak_f1 = -1
        min_second_f1 = -1
        peak_width_f1 = 0
    
    else:
        # The only peak is hidden
        main_peak_f1 = -1
        second_peak_f1 = -1
        min_second_f1 = -1
        peak_width_f1 = 0

    extrema_f1 = np.array([main_peak_f1, min_second_f1, second_peak_f1])

    # Filter 2
    peaks_max_f2, _ = find_peaks(flux_aug[f2_values_aug], prominence = 0.05)
    peaks_min_f2, _ = find_peaks(-flux_aug[f2_values_aug], prominence = 0.05)

    if len(peaks_max_f2) > 1:
        main_peak_f2, second_peak_f2 = find_main_and_second_peak(time[f2_values], time_aug[f2_values_aug], flux_aug[f2_values_aug], peaks_max_f2)

        min_distance_to_second_peak_f2 = np.abs(time_aug[f2_values_aug][peaks_min_f2] - time_aug[f2_values_aug][second_peak_f2])
        min_second_f2 = peaks_min_f2[np.argmin(min_distance_to_second_peak_f2)]

        peak_width_f2 = np.abs(time_aug[f2_values_aug][second_peak_f2] - time_aug[f2_values_aug][min_second_f2])

    elif len(peaks_max_f2) == 1 and len(peaks_min_f2) == 1:
        main_peak_f2 = -1
        second_peak_f2 = peaks_max_f2[0]
        min_second_f2 = peaks_min_f2[0]
        peak_width_f2 = np.abs(time_aug[f2_values_aug][second_peak_f2] - time_aug[f2_values_aug][min_second_f2])

    elif len(peaks_max_f2) == 1:
        # If only one maxima
        main_peak_f2 = peaks_max_f2[0]
        second_peak_f2 = -1
        min_second_f2 = -1
        peak_width_f2 = 0
    
    else:
        # The only peak is hidden
        main_peak_f2 = -1
        second_peak_f2 = -1
        min_second_f2 = -1
        peak_width_f2 = 0  

    extrema_f2 = np.array([main_peak_f2, min_second_f2, second_peak_f2])

    flux = flux * flux_max + flux_min
    fluxerr = fluxerr * flux_max
    flux_aug = flux_aug * flux_max + flux_min
    fluxerr_aug = fluxerr_aug * flux_max

    return extrema_f1, peak_width_f1, extrema_f2, peak_width_f2

### Plotting ###

def plot_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug, peak_time, extrema_f1, extrema_f2):

    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    # the extrema
    plt.scatter(time_aug[f1_values_aug][extrema_f1] + peak_time, flux_aug[f1_values_aug][extrema_f1], s = 250, marker = "*", c = "tab:green", edgecolors = "black", label = "Extrema r-band" , zorder = 10)

    # the original and augmented data
    plt.fill_between(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug] - fluxerr_aug[f1_values_aug], flux_aug[f1_values_aug] + fluxerr_aug[f1_values_aug], color = "tab:blue", alpha = 0.1)
    plt.errorbar(time[f1_values] + peak_time, flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = "tab:blue", label = "Band: r", zorder = 5)
    plt.errorbar(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug], yerr = fluxerr_aug[f1_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = "tab:blue", zorder = 5)

    # the extrema
    plt.scatter(time_aug[f2_values_aug][extrema_f2] + peak_time, flux_aug[f2_values_aug][extrema_f2], s = 250, marker = "*", c = "tab:red", edgecolors = "black", label = "Extrema g-band", zorder = 10)

    # the original and augmented data
    plt.fill_between(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug] - fluxerr_aug[f2_values_aug], flux_aug[f2_values_aug] + fluxerr_aug[f2_values_aug], color = "tab:orange", alpha = 0.1)
    plt.errorbar(time[f2_values] + peak_time, flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = "tab:orange", label = "Band: g", zorder = 5)
    plt.errorbar(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug], yerr = fluxerr_aug[f2_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = "tab:orange", zorder = 5)

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    # plt.savefig(f"ZTF_lightcurves_extrema_plots/ZTF_extrema_{SN_id}", dpi = 300)
    plt.show()

# %%%

########################################## PLOTTING ####################################################

def plot_best_fit_light_curve(SN_id, red_chi_squared, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, save_name):

    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    if f1 in filters:
        plt.plot(time_fit[f1_values_fit] + peak_time, flux_fit[f1_values_fit], linestyle = "--", linewidth = 2, alpha = 0.9, color = "tab:blue", label = f"Best-fitted light curve {f1}-band")
        plt.errorbar(time[f1_values] + peak_time, flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.3, color = "tab:blue", label = f"Band: {f1}", zorder = 5)

    if f2 in filters:
        plt.plot(time_fit[f2_values_fit] + peak_time, flux_fit[f2_values_fit], linestyle = "--", linewidth = 2, alpha = 0.9, color = "tab:orange", label = f"Best-fitted light curve {f2}-band")                
        plt.errorbar(time[f2_values] + peak_time, flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.3, color = "tab:orange", label = f"Band: {f2}", zorder = 5)

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}. " + r"$\mathrm{X}^{2}_{red}$" + f" = {red_chi_squared:.2f}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.savefig(f"Plots/Analytical_plots/{save_name}", dpi = 300)
    plt.close()

# %%

##################################### RESIDUAL / ONE-PEAK LIGHT CURVE ########################################

### Fitting functions ###

def calculate_flux_one_peak(time, cube, max_flux):

    estimated_flux = max_flux * cube[0] / (1 + np.exp(-(time - cube[1]) / cube[2]))
    
    time_constr = (time - cube[1] < cube[3])

    estimated_flux[time_constr] *= 1 - cube[4] * (time[time_constr] - cube[1])
    estimated_flux[~time_constr] *= (1 - cube[4] * cube[3]) * np.exp((cube[3] - (time[~time_constr] - cube[1])) / cube[5])

    return estimated_flux

def light_curve_one_peak(time, cube, max_flux, f1_values, f2_values):

    estimated_flux = []

    # Filter 1
    time_f1 = time[f1_values]
    cube_f1 = [10 ** cube[0], cube[1], 10 ** cube[2], 10 ** cube[3], cube[4], 10 **cube[5]]
    estimated_flux.extend(calculate_flux_one_peak(time_f1, cube_f1, max_flux))

    # Filter 2
    time_f2 = time[f2_values]
    cube_f2 = [10 ** (cube[0] + cube[7]), cube[1] + cube[8], 10 ** (cube[2] + cube[9]), 10 ** (cube[3] + cube[10]), \
                     cube[4] * (10 ** cube[11]), 10 ** (cube[5] + cube[12])]
    estimated_flux.extend(calculate_flux_one_peak(time_f2, cube_f2, max_flux))

    estimated_flux = np.array(estimated_flux)

    return estimated_flux

def function_approximation_one_peak(time, *cube):
    
    global flux, f1_values, f2_values, peak_main_idx
    
    cube = list(cube)
    return  light_curve_one_peak(time, cube, flux[peak_main_idx], f1_values, f2_values)

def reduced_chi_squared_one_peak(cube):
        
    global time, flux, fluxerr, f1_values, f2_values, peak_main_idx

    estimate = light_curve_one_peak(time, cube[0:14], flux[peak_main_idx], f1_values, f2_values)

    # error_param = 10 ** np.concatenate((np.full(len(f1_values[0]), cube[6]), np.full(len(f2_values[0]), cube[6] * cube[13])))
    error_squared = fluxerr ** 2 #+ error_param ** 2

    chi_squared = np.sum((flux - estimate) ** 2 / error_squared)

    return chi_squared / n_parameters_one_peak

def loglikelihood_one_peak(cube, ndim, nparams):

    return - 0.5 * reduced_chi_squared_one_peak(cube)

### Plotting ###

def plot_best_fit_light_curve_one_peak(SN_id, survey, red_chi_squared, time, flux, fluxerr, f1_values, f2_values, 
                                       amount_fit, time_fit, flux_fit, f1_values_fit, f2_values_fit, save_fig = False):
    
    global f1, f2, peak_main_idx
    
    if len(f1_values[0]) != 0:
        # Plot original data
        plt.errorbar(time[f1_values], flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = f"Samples {f1}-band")
        # Plot best fit
        plt.plot(time_fit[:amount_fit], flux_fit[:amount_fit], linestyle = "--", linewidth = 2, alpha = 0.9, color = "tab:blue", label = f"Best-fitted light curve {f1}-band")
        
    if len(f2_values[0]) != 0:
        # Plot original data
        plt.errorbar(time[f2_values], flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = f"Samples {f2}-band")
        # Plot best fit
        plt.plot(time_fit[amount_fit:], flux_fit[amount_fit:], linestyle = "--", linewidth = 2, alpha = 0.9, color = "tab:orange", label = f"Best-fitted light curve {f2}-band")
    
    # Plot posterior samples
    df = pd.read_csv(f"Data/Nested_sampling_parameters/{survey}/{SN_id}/one_peak/post_equal_weights.dat", delimiter = "\t")
    posterior_samples = df.to_numpy()  

    for idx in range(30):

        string = posterior_samples[idx][0]
        string_list = string.split()
        float_list = [float(num) for num in string_list]

        flux_fit = light_curve_one_peak(time_fit - time[peak_main_idx], float_list[:n_parameters_one_peak], flux[peak_main_idx], f1_values_fit, f2_values_fit)
        
        plt.plot(time_fit[:amount_fit], flux_fit[:amount_fit], linewidth = 1, alpha = 0.2, color = "tab:blue")
        plt.plot(time_fit[amount_fit:], flux_fit[amount_fit:], linewidth = 1, alpha = 0.2, color = "tab:orange")

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}. " + r"$\mathrm{X}^{2}_{red}$" + f" = {red_chi_squared:.2f}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/Nested_sampling_plots/{survey}/one_peak/LC_one_peak_{SN_id}", dpi = 300)
        plt.close()
    else:
        plt.show()

### Find best fitting parameters ###

def find_parameters_one_peak(SN_id, survey):

    global time, flux, fluxerr, f1_values, f2_values, mean_one_peak, std_one_peak

    # Calculate the reduced chi squared value of the intial guess
    red_chi_squared = reduced_chi_squared_one_peak(mean_one_peak) 

    done = True
    while done:

        # Folder in which sampling results will be stored
        os.makedirs(f"Data/Nested_sampling_parameters/{survey}/one_peak/{SN_id}/", exist_ok = True)
        
        # Run nested sampling to find the best parameters
        pymultinest.run(loglikelihood_one_peak, prior_one_peak, n_parameters_one_peak, n_live_points = 50,
                    outputfiles_basename = f"Data/Nested_sampling_parameters/{survey}/one_peak/{SN_id}/",
                    resume = False, verbose = True)
        
        # Retrieve the best parameters and their errors
        analyse = pymultinest.Analyzer(n_params = n_parameters_one_peak, outputfiles_basename = f"Data/Nested_sampling_parameters/{survey}/one_peak/{SN_id}/")
        parameter_info = analyse.get_stats()
        parameter_values = parameter_info["modes"][0]["mean"]
        parameter_sigma = parameter_info["modes"][0]["sigma"]

        # Calculate the reduced chi squared using the posterior
        df = pd.read_csv(f"Data/Nested_sampling_parameters/{survey}/one_peak/{SN_id}/post_equal_weights.dat", delimiter = "\t")
        posterior_samples = df.to_numpy()
        posterior_red_chi_squared = []       

        for idx in range(len(posterior_samples)):

            string = posterior_samples[idx][0]
            string_list = string.split()
            float_list = [float(num) for num in string_list]

            posterior_red_chi_squared.append(reduced_chi_squared_one_peak(float_list[:n_parameters_one_peak]))

        new_red_chi_squared = np.median(posterior_red_chi_squared)
    
        # Continue if the reduced chi squared has not converged yet
        if np.abs(red_chi_squared - new_red_chi_squared) > 0.001: # and new_red_chi_squared > 0.95:
            red_chi_squared = new_red_chi_squared
            mean_one_peak = parameter_values
            std_one_peak = parameter_sigma

        else:
            mean_one_peak = initial_mean_one_peak
            std_one_peak = initial_std_one_peak
            return np.array(parameter_values), np.array(parameter_sigma)
        
# %%

######################################## SECOND PEAK #########################################################

### Fit second peak ###

def gaussian_distribution(time, amplitude, mean, std):

    global flux, peak_main_idx
    
    return flux[peak_main_idx] * amplitude * np.exp(-(time - mean)**2 / (2 * std**2))

def second_peak(time, amplitude, mean, std, amplitude_ratio, mean_ratio, std_ratio):

    global f1_values_peak, f2_values_peak

    estimated_flux = []

    time_f1 = time[f1_values_peak]
    estimated_flux.extend(gaussian_distribution(time_f1, amplitude, mean, std))

    time_f2 = time[f2_values_peak]
    estimated_flux.extend(gaussian_distribution(time_f2, amplitude * amplitude_ratio, mean * mean_ratio, std * std_ratio))
    
    return np.array(estimated_flux)

def second_peak_fit(time, amplitude, mean, std, amplitude_ratio, mean_ratio, std_ratio):

    global f1_values, f2_values

    estimated_flux = []

    time_f1 = time[f1_values]
    estimated_flux.extend(gaussian_distribution(time_f1, amplitude, mean, std))

    time_f2 = time[f2_values]
    estimated_flux.extend(gaussian_distribution(time_f2, amplitude * amplitude_ratio, mean * mean_ratio, std * std_ratio))
    
    return np.array(estimated_flux)

### Plotting ###

def plot_extrema_and_second_peak(SN_id, time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug, 
                                 time_fit, flux_fit, f1_values_fit, f2_values_fit, extrema_f1, extrema_f2, peak_time):

    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    if f1 in filters:
        # the extrema
        # plt.scatter(time_aug[f1_values_aug][extrema_f1][::2] + peak_time, flux_aug[f1_values_aug][extrema_f1][::2], s = 250, marker = "*", c = "tab:green", edgecolors = "black", label = "Extrema r-band" , zorder = 10)
        # plt.scatter(time_aug[f1_values_aug][extrema_f1][1] + peak_time, flux_aug[f1_values_aug][extrema_f1][1], s = 250, marker = "*", c = "tab:green", edgecolors = "black", zorder = 10)

        # the second peak
        plt.plot(time_fit[f1_values_fit] + peak_time, flux_fit[f1_values_fit], color = "tab:blue")

        # the original and augmented data
        plt.fill_between(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug] - fluxerr_aug[f1_values_aug], flux_aug[f1_values_aug] + fluxerr_aug[f1_values_aug], color = "tab:blue", alpha = 0.1)
        plt.errorbar(time[f1_values] + peak_time, flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = "tab:blue", label = "Band: r", zorder = 5)
        plt.errorbar(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug], yerr = fluxerr_aug[f1_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = "tab:blue", zorder = 5)

    if f2 in filters:
        # the extrema
        # plt.scatter(time_aug[f2_values_aug][extrema_f2][::2] + peak_time, flux_aug[f2_values_aug][extrema_f2][::2], s = 250, marker = "*", c = "tab:red", edgecolors = "black", label = "Extrema g-band", zorder = 10)
        # plt.scatter(time_aug[f2_values_aug][extrema_f2][1] + peak_time, flux_aug[f2_values_aug][extrema_f2][1], s = 250, marker = "*", c = "tab:red", edgecolors = "black", zorder = 10)

        # the second peak
        plt.plot(time_fit[f2_values_fit] + peak_time, flux_fit[f2_values_fit], color = "tab:orange")

        # the original and augmented data
        plt.fill_between(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug] - fluxerr_aug[f2_values_aug], flux_aug[f2_values_aug] + fluxerr_aug[f2_values_aug], color = "tab:orange", alpha = 0.1)
        plt.errorbar(time[f2_values] + peak_time, flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = "tab:orange", label = "Band: g", zorder = 5)
        plt.errorbar(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug], yerr = fluxerr_aug[f2_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = "tab:orange", zorder = 5)

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    # plt.savefig(f"Plots/ZTF_lightcurves_extrema_plots/ZTF_extrema_{SN_id}", dpi = 300)
    plt.show()

### Isolate peak ### 

def isolate_second_peak(main_filter_second_peak, extrema_f1, peak_width_f1, extrema_f2, peak_width_f2, 
                        time_aug, flux_aug, fluxerr_aug, filters_aug, f1_values_aug, f2_values_aug):

    global time, flux, fluxerr, filters, f1, f2

    # Isolate the original data points that are part of the peak
    if main_filter_second_peak == f1:
        peak_data = np.where(np.abs(time - time_aug[f1_values_aug][extrema_f1][2]) < peak_width_f1)

    elif main_filter_second_peak == f2:
        peak_data = np.where(np.abs(time - time_aug[f2_values_aug][extrema_f2][2]) < peak_width_f2)

    # If the peak contains more data points than the number of fit parameters
    # if len(peak_data[0]) > 6:

    time_peak = time[peak_data]
    flux_peak = flux[peak_data]
    fluxerr_peak = fluxerr[peak_data]
    filters_peak = filters[peak_data]

    # If it contains less data points, use the augmented data
    # else:
        # if main_filter_second_peak == "r":
        #     peak_data = np.where(np.abs(time_aug - time_aug[f1_values_aug][extrema_f1][2]) < peak_width_f1)

        # elif main_filter_second_peak == "g":
        #     peak_data = np.where(np.abs(time_aug - time_aug[f2_values_aug][extrema_f2][2]) < peak_width_f2)

        # time_peak = time_aug[peak_data]
        # flux_peak = flux_aug[peak_data]
        # fluxerr_peak = fluxerr_aug[peak_data]
        # filters_peak = filters_aug[peak_data]

    return time_peak, flux_peak, fluxerr_peak, filters_peak

# %%

######################################## FULL LIGHT CURVE ###########################################################

def calculate_flux_two_peaks(time, cube, max_flux):

    estimated_flux = max_flux * cube[0] / (1 + np.exp(-(time - cube[1]) / cube[2]))
    
    time_constr = (time - cube[1] < cube[3])

    estimated_flux[time_constr] *= 1 - cube[4] * (time[time_constr] - cube[1])
    estimated_flux[~time_constr] *= (1 - cube[4] * cube[3]) * np.exp((cube[3] - (time[~time_constr] - cube[1])) / cube[5])

    estimated_flux += max_flux * cube[6] * np.exp(-(time - cube[7])**2 / (2 * cube[8]**2))

    return estimated_flux

def light_curve_two_peaks(time, cube, max_flux, f1_values, f2_values):

    estimated_flux = []

    # Filter 1
    time_f1 = time[f1_values]
    cube_f1 = [10 ** cube[0], cube[1], 10 ** cube[2], 10 ** cube[3], cube[4], 10 **cube[5], cube[14], cube[15], cube[16]]
    estimated_flux.extend(calculate_flux_two_peaks(time_f1, cube_f1, max_flux))

    # Filter 2
    time_f2 = time[f2_values]
    cube_f2 = [10 ** (cube[0] + cube[7]), cube[1] + cube[8], 10 ** (cube[2] + cube[9]), 10 ** (cube[3] + cube[10]), \
                     cube[4] * (10 ** cube[11]), 10 ** (cube[5] + cube[12]), \
                     cube[14] * cube[17], cube[15] * cube[18], cube[16] * cube[19]]
    estimated_flux.extend(calculate_flux_two_peaks(time_f2, cube_f2, max_flux))

    estimated_flux = np.array(estimated_flux)

    return estimated_flux

def function_approximation_two_peaks(time, *cube):

    global flux, f1_values, f2_values, peak_main_idx

    cube = list(cube)
    return light_curve_two_peaks(time, cube, flux[peak_main_idx], f1_values, f2_values)

def function_approximation_two_peaks_fit(time, f1_values_fit, f2_values_fit, *cube):

    global flux, peak_main_idx

    cube = list(cube)
    return light_curve_two_peaks(time, cube, flux[peak_main_idx], f1_values_fit, f2_values_fit)

def reduced_chi_squared_two_peaks(cube):
        
    global time, flux, fluxerr, f1_values, f2_values, peak_main_idx

    estimate = light_curve_two_peaks(time, cube, flux[peak_main_idx], f1_values, f2_values)

    # error_param = 10 ** np.concatenate((np.full(len(f1_values[0]), cube[6]), np.full(len(f2_values[0]), cube[6] * cube[13])))
    error_squared = fluxerr ** 2 #+ error_param ** 2

    chi_squared = np.sum((flux - estimate) ** 2 / error_squared)

    return chi_squared / n_parameters_two_peaks

def loglikelihood_two_peaks(cube, ndim, nparams):

    return - 0.5 * reduced_chi_squared_two_peaks(cube)

def find_parameters_two_peaks(SN_id, survey):

    global time, flux, fluxerr, f1_values, f2_values, mean_two_peaks, std_two_peaks 

    # Calculate the reduced chi squared value of the intial guess
    red_chi_squared = reduced_chi_squared_two_peaks(mean_two_peaks) 

    done = True
    while done:

        # Folder in which sampling results will be stored
        os.makedirs(f"Data/Nested_sampling_parameters/{survey}/two_peaks/{SN_id}/", exist_ok = True)
        
        # Run nested sampling to find the best parameters
        pymultinest.run(loglikelihood_two_peaks, prior_two_peaks, n_parameters_two_peaks, n_live_points = 50,
                    outputfiles_basename = f"Data/Nested_sampling_parameters/{survey}/two_peaks/{SN_id}/",
                    resume = False, verbose = True)
        
        # Retrieve the best parameters and their errors
        analyse = pymultinest.Analyzer(n_params = n_parameters_two_peaks, outputfiles_basename = f"Data/Nested_sampling_parameters/{survey}/two_peaks/{SN_id}/")
        parameter_info = analyse.get_stats()
        parameter_values = parameter_info["modes"][0]["mean"]
        parameter_sigma = parameter_info["modes"][0]["sigma"]

        # Calculate the reduced chi squared using the posterior
        df = pd.read_csv(f"Data/Nested_sampling_parameters/{survey}/two_peaks/{SN_id}/post_equal_weights.dat", delimiter = "\t")
        posterior_samples = df.to_numpy()
        posterior_red_chi_squared = []       

        for idx in range(len(posterior_samples)):

            string = posterior_samples[idx][0]
            string_list = string.split()
            float_list = [float(num) for num in string_list]

            posterior_red_chi_squared.append(reduced_chi_squared_two_peaks(float_list[:n_parameters_two_peaks]))

        new_red_chi_squared = np.median(posterior_red_chi_squared)
    
        # Continue if the reduced chi squared has not converged yet
        if np.abs(red_chi_squared - new_red_chi_squared) > 0.001: # and new_red_chi_squared > 0.95:
            red_chi_squared = new_red_chi_squared
            mean_two_peaks = parameter_values
            std_two_peaks = parameter_sigma

        else:
            mean_two_peaks = initial_mean_two_peaks
            std_two_peaks = initial_std_two_peaks
            return np.array(parameter_values), np.array(parameter_sigma)

# %% 

def fit_light_curve(SN_id, survey):

    print(SN_id)

    global mean_one_peak, std_one_peak, mean_two_peaks, std_two_peaks, parameter_bounds_one_peak, parameter_bounds_gaussian
    global f1, f2
    global time, flux, fluxerr, filters, f1_values, f2_values, peak_main_idx, f1_values_peak, f2_values_peak, main_flux_second_peak

    ########################################## PRE-PROCESSING ####################################################

    ###### Retrieve data ######
    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        # Load the data
        time, flux, fluxerr, filters, boundaries = ztf_load_data(SN_id)

    if survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, fluxerr, filters, boundaries = atlas_load_data(SN_id)

    # Only consider data belonging to the supernova explosion
    time = np.concatenate((time[boundaries[0] : boundaries[2] + 1], time[boundaries[1] : boundaries[3] + 1]))
    flux = np.concatenate((flux[boundaries[0] : boundaries[2] + 1], flux[boundaries[1] : boundaries[3] + 1]))
    fluxerr = np.concatenate((fluxerr[boundaries[0] : boundaries[2] + 1], fluxerr[boundaries[1] : boundaries[3] + 1]))
    filters = np.concatenate((filters[boundaries[0] : boundaries[2] + 1], filters[boundaries[1] : boundaries[3] + 1]))

    # Only consider confident detections
    confident_detections = flux/fluxerr > 3

    time = time[confident_detections]
    flux = flux[confident_detections]
    fluxerr = fluxerr[confident_detections]
    filters = filters[confident_detections]

    f1_values = np.where(filters == f1)
    f2_values = np.where(filters == f2)

    # Shift the light curve so that the main peak is at time = 0 MJD
    if f1 in filters:
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])
        
    time -= peak_time

    ###### Augment data ######
    amount_aug = 70

    time_aug, flux_aug, fluxerr_aug, filters_aug = augment_data(survey, peak_time, amount_aug)

    f1_values_aug = np.where(filters_aug == f1)
    f2_values_aug = np.where(filters_aug == f2)

    ###### Find extrema ######
    extrema_f1, peak_width_f1, extrema_f2, peak_width_f2 = find_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug)

    plot_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug, peak_time, extrema_f1, extrema_f2)

    ########################################## LIGHT CURVE FITTING ####################################################

    # # Data for plotting
    # amount_fit = 200

    # time_fit = np.concatenate((np.linspace(time.min(), time.max(), amount_fit), np.linspace(time.min(), time.max(), amount_fit)))
    # f1_values_fit = np.arange(amount_fit)
    # f2_values_fit = np.arange(amount_fit) + amount_fit

    # if extrema_f1[2] == -1 and extrema_f2[2] == -1:
    #     # Only a one-peak light curve can be fit through the data

    #     # Nested sampling
    #     one_peak_parameters, one_peak_uncertainties = find_parameters_one_peak(survey, SN_id)
    #     parameter_bounds = (one_peak_parameters - 3 * one_peak_uncertainties, one_peak_parameters + 3 * one_peak_uncertainties)

    #     # Levenberg-Marquardt algorithm 
    #     try:
    #         imporved_peak_parameters, _ = curve_fit(function_approximation_one_peak, time, flux, p0 = one_peak_parameters, sigma = fluxerr, 
    #                                                 bounds = parameter_bounds, maxfev = 100000)
        
    #     except RuntimeError:
    #         imporved_peak_parameters = one_peak_parameters

    #     # Plot the results
    #     np.save(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/parameters_{SN_id}_OP", imporved_peak_parameters)
    #     red_chi_squared = reduced_chi_squared_one_peak(imporved_peak_parameters)

    #     with open(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
    #         writer = csv.writer(file)
    #         writer.writerow([SN_id, red_chi_squared])
        
    #     flux_fit = light_curve_one_peak(time_fit, imporved_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
    #     plot_best_fit_light_curve(SN_id, red_chi_squared, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/forced_photometry/one_peak/light_curve_{SN_id}_OP")

    # else: 
    #     # Possible that there is a second peak

    #     ### Fit a one-peak light curve
    #     # Nested sampling
    #     one_peak_parameters, one_peak_uncertainties = find_parameters_one_peak(survey, SN_id)
    #     parameter_bounds = (one_peak_parameters - 3 * one_peak_uncertainties, one_peak_parameters + 3 * one_peak_uncertainties)

    #     try:
    #         # Nested sampling + Levenberg-Marquardt algorithm 
    #         imporved_peak_parameters, _ = curve_fit(function_approximation_one_peak, time, flux, p0 = one_peak_parameters, sigma = fluxerr, 
    #                                                 bounds = parameter_bounds, maxfev = 100000)
        
    #     except RuntimeError:
    #         imporved_peak_parameters = one_peak_parameters
        
    #     red_chi_squared_OP = reduced_chi_squared_one_peak(imporved_peak_parameters)
               
    #     ## Fit a two-peaks light curve
    #     # Identify the main filter for the second peak ("r" if it exists and contains a peak)
    #     if f1 in filters and extrema_f1[2] != -1:
    #         main_filter_second_peak = f1

    #     elif f2 in filters and extrema_f2[2] != -1:
    #         main_filter_second_peak = f2

    #     time_peak, flux_peak, fluxerr_peak, filters_peak = isolate_second_peak(main_filter_second_peak, extrema_f1, peak_width_f1, 
    #                                                                            extrema_f2, peak_width_f2, time_aug, flux_aug, 
    #                                                                            fluxerr_aug, filters_aug, f1_values_aug, f2_values_aug)

    #     if len(time_peak) < 6:
    #          red_chi_squared_TP = np.inf

    #     else:
    #         f1_values_peak = np.where(filters_peak == f1)
    #         f2_values_peak = np.where(filters_peak == f2)

    #         # Inital guess of the parameters (amplitude, mean, standard deviation)
    #         # If in both filters a second peak is detected
    #         if f1 in filters and extrema_f1[2] != -1 and f2 in filters and extrema_f2[2] != -1:
    #             guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 
    #                                         flux_aug[f2_values_aug][extrema_f2][2] / flux_aug[f1_values_aug][extrema_f1][2], 
    #                                         time_aug[f2_values_aug][extrema_f2][2] / time_aug[f1_values_aug][extrema_f1][2], 
    #                                         peak_width_f2 / peak_width_f1])
            
    #         # If only in one peak a second peak is detected
    #         else:
    #             if f1 in filters and extrema_f1[2] != -1:
    #                 guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 1, 1, 1])

    #             elif f2 in filters and extrema_f2[2] != -1:
    #                 guess_parameters = np.array([1, time_aug[f2_values_aug][extrema_f2][2], peak_width_f2, 1, 1, 1])

    #         down_bound = [0, -300, 0, -0.5, -1.5, -1.5]
    #         up_bound = [1.5, 300, 250, 1.5, 5.0, 5.0]

    #         if not ((down_bound < guess_parameters) & (guess_parameters < up_bound)).all():
    #             # The one-peak light curve is a better fit
    #             np.save(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/parameters_{SN_id}_OP", imporved_peak_parameters)
                
    #             with open(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
    #                 writer = csv.writer(file)
    #                 writer.writerow([SN_id, red_chi_squared_OP])

    #             # Plot the results
    #             flux_fit = light_curve_one_peak(time_fit, imporved_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
    #             plot_best_fit_light_curve(SN_id, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/forced_photometry/one_peak/light_curve_{SN_id}_OP")

    #             return 
            
    #         else:

    #             # Estimate the parameters of the second peak in both bands simultaneously
    #             try:
    #                 second_peak_parameters, _ = curve_fit(second_peak, time_peak, flux_peak, p0 = guess_parameters, sigma = fluxerr_peak, bounds = parameter_bounds_gaussian, maxfev = 100000)
                
    #             except RuntimeError:
    #                 second_peak_parameters = guess_parameters

    #             except ValueError:
    #                 # The one-peak light curve is a better fit
    #                 np.save(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/parameters_{SN_id}_OP", imporved_peak_parameters)

    #                 with open(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
    #                     writer = csv.writer(file)
    #                     writer.writerow([SN_id, red_chi_squared_OP])

    #                 # Plot the results
    #                 flux_fit = light_curve_one_peak(time_fit, imporved_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
    #                 plot_best_fit_light_curve(SN_id, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/forced_photometry/one_peak/light_curve_{SN_id}_OP")

    #                 return

    #             # Calculate the predicted flux of the second peak
    #             flux_peak_fit = second_peak_fit(time, *second_peak_parameters)

    #             # Calculate the flux without the second peak
    #             original_flux = np.copy(flux)
    #             flux -= flux_peak_fit

    #             residual_parameters, _ = find_parameters_one_peak(survey, SN_id)

    #             flux = original_flux

    #             initial_guesses = list(residual_parameters) + list(second_peak_parameters)

    #             # Estimate the parameters of the full light curve in both bands simultaneously
    #             try:
    #                 two_peaks_parameters, _ = curve_fit(function_approximation_two_peaks, time, flux, p0 = initial_guesses, sigma = fluxerr, bounds = parameter_bounds_two_peaks, maxfev = 100000)
                
    #             except RuntimeError:
    #                 two_peaks_parameters = initial_guesses

    #             red_chi_squared_TP = reduced_chi_squared_two_peaks(two_peaks_parameters)

    #             ### Based on best red chi squared value save one or two peak parameter and plot

    #     if np.abs(red_chi_squared_OP - 1) < np.abs(red_chi_squared_TP - 1):
    #         # The one-peak light curve is a better fit
    #         np.save(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/parameters_{SN_id}_OP", imporved_peak_parameters)
            
    #         with open(f"Data/Analytical_parameters/{survey}/forced_photometry/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
    #             writer = csv.writer(file)
    #             writer.writerow([SN_id, red_chi_squared_OP])

    #         # Plot the results
    #         flux_fit_OP = light_curve_one_peak(time_fit, imporved_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
    #         plot_best_fit_light_curve(SN_id, red_chi_squared_OP, time_fit, flux_fit_OP, f1_values_fit, f2_values_fit, peak_time, f"{survey}/forced_photometry/one_peak/light_curve_{SN_id}_OP")

    #     else:
    #         # The two-peaks light curve is a better fit
    #         np.save(f"Data/Analytical_parameters/{survey}/forced_photometry/two_peaks/parameters_{SN_id}_TP", two_peaks_parameters) 

    #         with open(f"Data/Analytical_parameters/{survey}/forced_photometry/two_peaks/red_chi_squared_TP.csv", "a", newline = "") as file:
    #             writer = csv.writer(file)
    #             writer.writerow([SN_id, red_chi_squared_TP])

    #         # Plot the results
    #         flux_fit_TP = light_curve_two_peaks(time_fit, two_peaks_parameters, peak_flux, f1_values_fit, f2_values_fit)
    #         plot_best_fit_light_curve(SN_id, red_chi_squared_TP, time_fit, flux_fit_TP, f1_values_fit, f2_values_fit, peak_time, f"{survey}/forced_photometry/two_peaks/light_curve_{SN_id}_TP")

# %%

if __name__ == '__main__':
    
    survey = "ZTF"

    for SN_id in ["ZTF18aaxjuwy", "ZTF23aansdlc", "ZTF23aamanim", "ZTF19aasekcx", "ZTF20aadxrvb"]:

        fit_light_curve(SN_id, survey)

# %%
