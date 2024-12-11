# %%

__ImportError__ = "One or more required packages are not installed. See requirements.txt."

try:
    from data_processing import ztf_load_data, atlas_load_data

    import fulu
    from sklearn.gaussian_process.kernels import (RBF, Matern, 
        WhiteKernel, ConstantKernel as C)
    import pymultinest

    from scipy.stats import truncnorm
    from scipy.signal import find_peaks, peak_prominences, peak_widths
    from scipy.optimize import curve_fit

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import multiprocessing
    import os
    import csv

except ImportError:
    raise ImportError(__ImportError__)

plt.rcParams["text.usetex"] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

global parameter_bounds_one_peak, parameter_bounds_gaussian, parameter_bounds_two_peaks
global f1, f2
global time, flux, fluxerr, filters, f1_values, f2_values, peak_main_idx, f1_values_peak, f2_values_peak

# Parameters of a light curve with one peak
parameters_one_peak = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                       "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"]
n_parameters_one_peak = len(parameters_one_peak) 

# Parameters of a light curve with two peaks
parameters_two_peaks = parameters_one_peak + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]
n_parameters_two_peaks = len(parameters_two_peaks) 

# %%

# Initial guess of the values of the parameters
# Mean and std values of the posteriors found in the Superphot+ paper
guess_mean_one_peak = [0.0957, -17.878, 0.6664, 1.4258, 0.00833, 1.5261, -1.6629,  -0.0766, -3.4065, -0.1510, -0.0452, -0.2068, -0.1486, -0.1509] 
guess_std_one_peak = [0.0575, 9.916, 0.4250, 0.5079, 0.00585, 0.5057, 0.5578, 0.1096, 4.566, 0.1927, 0.1705, 0.2704, 0.2642, 0.2542] 

parameter_bounds_one_peak = ([-0.3, -100, -2, 0, 0, 0, -3, -1, -50, -1.5, -1.5, -2, -1.5, -1.5], [0.5, 30, 4, 3.5, 0.03, 4, -0.8, 1, 30, 1.5, 1.5, 1, 1.5, -1])

parameter_bounds_gaussian = ([0, -500, 0, -0.75, -0.75, -0.75], [1, 500, 150, 1.5, 1.5, 2.5])

parameter_bounds_two_peaks = (parameter_bounds_one_peak[0] + parameter_bounds_gaussian[0], parameter_bounds_one_peak[1] + parameter_bounds_gaussian[1])

def prior_one_peak(cube):

    """ 
    Transform the priors of the parameters from a uniform prior in the interval [0, 1] to the truncated Gaussian prior found in the Superphot+ paper

    Parameters:
        cube (list): List of parameters of the fitting model 

    Outputs: 
        cube (list): List of parameters of the fitting model, transformed
    """

    # A f1
    min = (-0.3 - guess_mean_one_peak[0]) / guess_std_one_peak[0]
    max = (0.5 - guess_mean_one_peak[0]) / guess_std_one_peak[0]
    cube[0] = truncnorm.ppf(cube[0], min, max, guess_mean_one_peak[0], guess_std_one_peak[0])

    # t_0 f1
    min = (-100.0 - guess_mean_one_peak[1]) / guess_std_one_peak[1]
    max = (30.0 - guess_mean_one_peak[1]) / guess_std_one_peak[1]
    cube[1] = truncnorm.ppf(cube[1], min, max, guess_mean_one_peak[1], guess_std_one_peak[1])

    # t_rise f1
    min = (-2.0 - guess_mean_one_peak[2]) / guess_std_one_peak[2]
    max = (4.0 - guess_mean_one_peak[2]) / guess_std_one_peak[2]
    cube[2] = truncnorm.ppf(cube[2], min, max, guess_mean_one_peak[2], guess_std_one_peak[2])

    # gamma f1
    min = (0.0 - guess_mean_one_peak[3]) / guess_std_one_peak[3]
    max = (3.5 - guess_mean_one_peak[3]) / guess_std_one_peak[3]
    cube[3] = truncnorm.ppf(cube[3], min, max, guess_mean_one_peak[3], guess_std_one_peak[3])

    # beta f1
    min = (0.0 - guess_mean_one_peak[4]) / guess_std_one_peak[4]
    max = (0.03 - guess_mean_one_peak[4]) / guess_std_one_peak[4]
    cube[4] = truncnorm.ppf(cube[4], min, max, guess_mean_one_peak[4], guess_std_one_peak[4])

    # t_fall f1
    min = (0.0 - guess_mean_one_peak[5]) / guess_std_one_peak[5]
    max = (4.0 - guess_mean_one_peak[5]) / guess_std_one_peak[5]
    cube[5] = truncnorm.ppf(cube[5], min, max, guess_mean_one_peak[5], guess_std_one_peak[5])

    # error f1
    min = (-3.0 - guess_mean_one_peak[6]) / guess_std_one_peak[6]
    max = (-0.8 - guess_mean_one_peak[6]) / guess_std_one_peak[6]
    cube[6] = truncnorm.ppf(cube[6], min, max, guess_mean_one_peak[6], guess_std_one_peak[6])

    # A f2
    min = (-1.0 - guess_mean_one_peak[7]) / guess_std_one_peak[7]
    max = (1.0 - guess_mean_one_peak[7]) / guess_std_one_peak[7]
    cube[7] = truncnorm.ppf(cube[7], min, max, guess_mean_one_peak[7], guess_std_one_peak[7])

    # t_0 f2
    min = (-50.0 - guess_mean_one_peak[8]) /guess_std_one_peak[8] 
    max = (30.0 - guess_mean_one_peak[8]) / guess_std_one_peak[8]
    cube[8] = truncnorm.ppf(cube[8], min, max, guess_mean_one_peak[8], guess_std_one_peak[8])

    # t_rise f2
    min = (-1.5 - guess_mean_one_peak[9]) / guess_std_one_peak[9]
    max = (1.5 - guess_mean_one_peak[9]) / guess_std_one_peak[9]
    cube[9] = truncnorm.ppf(cube[9], min, max, guess_mean_one_peak[9], guess_std_one_peak[9])

    # gamma f2
    min = (-1.5 - guess_mean_one_peak[10]) / guess_std_one_peak[10]
    max = (1.5 - guess_mean_one_peak[10]) / guess_std_one_peak[10]
    cube[10] = truncnorm.ppf(cube[10], min, max, guess_mean_one_peak[10], guess_std_one_peak[10])

    # beta f2
    min = (-2.0 - guess_mean_one_peak[11]) / guess_std_one_peak[11]
    max = (1.0 - guess_mean_one_peak[11]) / guess_std_one_peak[11]
    cube[11] = truncnorm.ppf(cube[11], min, max, guess_mean_one_peak[11], guess_std_one_peak[11])

    # t_fall f2
    min = (-1.5 - guess_mean_one_peak[12]) / guess_std_one_peak[12]
    max = (1.5 - guess_mean_one_peak[12]) / guess_std_one_peak[12]
    cube[12] = truncnorm.ppf(cube[12], min, max, guess_mean_one_peak[12], guess_std_one_peak[12])

    # error f2
    min = (-1.5 - guess_mean_one_peak[13]) / guess_std_one_peak[13]
    max = (-1.0 - guess_mean_one_peak[13]) / guess_std_one_peak[13]
    cube[13] = truncnorm.ppf(cube[13], min, max, guess_mean_one_peak[13], guess_std_one_peak[13])

    return cube

# %%

########################################## PRE-PROCESSING ####################################################

# %%

### Light curve approximation ###

def fulu_data_fitting(survey, time, flux, fluxerr, filters, augmentation_type):

    """ 
    Fit the light curve data in every filter using the fulu pacakge

    Parameters:
        survey (str): Survey that observed the SN 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observations
        augmentation_type (str): Technique used to augment the data: GP, MLP, NF, BNN

    Outputs: 
        augmentation (object): Object fitted by the light curve data
    """

    try:
        if survey == "ZTF":
            passbands_wavelength = {'r': np.log10(6366.38), 'g': np.log10(4746.48)}

        elif survey == "ATLAS":
            passbands_wavelength = {'o': np.log10(6629.82), 'c': np.log10(5182.42)}

    except:
        print("ERROR: the options for survey are \"ZTF\" and \"ATLAS\".")

    try:
        if augmentation_type == "GP":
            augmentation = fulu.GaussianProcessesAugmentation(passbands_wavelength, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())

        elif augmentation_type == "MLP":
            augmentation = fulu.MLPRegressionAugmentation(passbands_wavelength)
        
        elif augmentation_type == "NF":
            augmentation = fulu.NormalizingFlowAugmentation(passbands_wavelength)
        
        elif augmentation_type == "BNN":
            augmentation = fulu.BayesianNetAugmentation(passbands_wavelength)
    
    except:
        print("ERROR: the options for augmentation_type are \"GP\", \"MLP\", \"NF\"and \"BNN\".")

    augmentation.fit(time, flux, fluxerr, filters)

    return augmentation

### Data augmentation ###

def augment_data(survey, amount_augmentation):

    """ 
    Augment the number of observations in every filter using the fulu package 

    Parameters:
        survey (str): Survey that observed the SN 
        amount_augmentation (int): Number of augmented observations in one filter

    Outputs: 
        time_aug (numpy.ndarray): Modified Julian Date of the augmented observations 
        flux_aug (numpy.ndarray): Differential flux of the augmented observations in microJansky
        fluxerr_aug (numpy.ndarray): One-sigma error on the differential flux of the augmented observations in microJansky
        filters_aug (numpy.ndarray): Filter of the augmented observations
    """

    global time, flux, fluxerr, filters, f1_values, f2_values

    # Augment the amount of data points using Gaussian Processing (fulu)
    augmentation = fulu_data_fitting(survey, time, flux, fluxerr, filters, augmentation_type = "GP")

    time_aug, flux_aug, fluxerr_aug, filters_aug = augmentation.augmentation(time.min(), time.max(), n_obs = amount_augmentation)

    f1_values_aug = np.where(filters_aug == f1)
    f2_values_aug = np.where(filters_aug == f2)

    # Remove the augmented data after the last datapoint in each filter
    # Extrapolated data can lead to fake additional peaks
    if f1 in filters:
        out_of_range_f1 = np.where((time_aug[f1_values_aug] > time[f1_values].max()))[0]
    else:
        out_of_range_f1 = np.array([])

    if f2 in filters:
        out_of_range_f2 = np.where((time_aug[f2_values_aug] > time[f2_values].max()))[0] + amount_augmentation
    else:
        out_of_range_f2 = np.array([])

    if len(np.concatenate((out_of_range_f1, out_of_range_f2))) != 0 :
        time_aug = np.delete(time_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        flux_aug = np.delete(flux_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        fluxerr_aug = np.delete(fluxerr_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))
        filters_aug = np.delete(filters_aug, np.concatenate((out_of_range_f1, out_of_range_f2)))

    return time_aug, flux_aug, fluxerr_aug, filters_aug

### Find peaks ###

def find_main_and_second_peak(time, time_aug, flux_aug, peaks_max):

    """ 
    Determine the indices of the two largest flux maxima

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the processed observations 
        time_aug (numpy.ndarray): Modified Julian Date of the augmented observations 
        flux_aug (numpy.ndarray): Differential flux of the augmented observations in microJansky
        peaks_max (list): Indices of all light curve maxima

    Outputs: 
        main_peak (int): Index of the main flux peak
        second_peak (int): Index of the second largest flux peak
    """

    # Calculate prominences
    prominences = peak_prominences(flux_aug, peaks_max)[0]

    # Calculate peak widths in number of data points (full width at half maximum)
    widths = peak_widths(flux_aug, peaks_max, rel_height = 0.5)[0]
    number_data_points = [len(np.where(np.abs(time - time_aug[peaks_max][idx]) < widths[idx])[0]) \
                        for idx in range(len(peaks_max))]

    # Calculate peak heights
    peak_heights = flux_aug[peaks_max]

    peak_info = list(zip(peaks_max, peak_heights, prominences, number_data_points))

    sorted_by_height = sorted(peak_info, key=lambda x: x[1], reverse=True)
    main_peak = sorted_by_height[0]

    second_largest_peak_candidates = [peak for peak in peak_info if peak != main_peak]

    # Sort by prominence, number of data points (widths), and height
    sorted_candidates = sorted(second_largest_peak_candidates, key = lambda x: (x[2], x[3], x[1]), reverse = True)
    second_peak = sorted_candidates[-1]

    return main_peak[0], second_peak[0]

def find_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug):

    """ 
    Determine the extrema (two maxima and minimum in between) and width of the second peak of the augmentated data in every filter

    Parameters:
        time_aug (numpy.ndarray): Modified Julian Date of the augmented observations 
        flux_aug (numpy.ndarray): Differential flux of the augmented observations in microJansky
        fluxerr_aug (numpy.ndarray): One-sigma error on the differential flux of the augmented observations in microJansky
        f1_values_aug (numpy.ndarray): Indices of augmented observations in the f1 filter
        f2_values_aug (numpy.ndarray): Indices of augmented observations in the f2 filter

    Outputs: 
        extrema_f1 (numpy.ndarray): Indices of the extrema in the f1 filter
        peak_width_f1 (float): Width of the second peak in the f1 filter
        extrema_f2 (numpy.ndarray): Indices of the extrema in the f2 filter
        peak_width_f2 (float): Width of the second peak in the f2 filter
    """

    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    # Reshape the data so that the flux is between 0 and 1 microJansky
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

        # Find the minimum closest to the second peak 
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
        main_peak_f2 = peaks_max_f2[0]
        second_peak_f2 = -1
        min_second_f2 = -1
        peak_width_f2 = 0
    
    else:
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

# %%

##################################### RESIDUAL / ONE-PEAK LIGHT CURVE ########################################

### Fitting functions ###

def calculate_flux_one_peak(time, cube, max_flux):

    """ 
    The one-peak fitting model from Superphot+ for observations in one filter

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the observations in a certain filter 
        cube (list): List of (estimated) parameters of the fitting model in a certain filter 
        max_flux (float): Maximum flux in the f1 filter 

    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations in a certain filter 
    """

    # Exponential rise in brightness
    estimated_flux = max_flux * cube[0] / (1 + np.exp(-(time - cube[1]) / cube[2]))
    
    time_plateau = (time - cube[1] < cube[3])

    # Linear plateau after peak
    estimated_flux[time_plateau] *= 1 - cube[4] * (time[time_plateau] - cube[1])

    # Exponential decline in brightness
    estimated_flux[~time_plateau] *= (1 - cube[4] * cube[3]) * np.exp((cube[3] - (time[~time_plateau] - cube[1])) / cube[5])

    return estimated_flux

def light_curve_one_peak(time, cube, max_flux, f1_values, f2_values):

    """ 
    The one-peak fitting model from Superphot+ for observations in both filters

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the observations 
        cube (list): List of (estimated) parameters of the fitting model 
        max_flux (float): Maximum flux in the f1 filter 
        f1_values (numpy.ndarray): Indices of observations in the f1 filter
        f2_values (numpy.ndarray): Indices of observations in the f2 filter

    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations
    """

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

def reduced_chi_squared_one_peak(cube):

    """ 
    Calculate the reduced chi-squared value for estimated parameters of the one-peak model

    Parameters:
        cube (list): List of (estimated) parameters of the one-peak fitting model 

    Outputs: 
        red_chi_squared (float): Reduced chi-squared value
    """
        
    global time, flux, fluxerr, f1_values, f2_values, peak_main_idx

    estimated_flux = light_curve_one_peak(time, cube[0:14], flux[peak_main_idx], f1_values, f2_values)

    error_param = 10 ** np.concatenate((np.full(len(f1_values[0]), cube[6]), np.full(len(f2_values[0]), cube[6] * cube[13])))
    error_squared = fluxerr ** 2 + error_param ** 2

    chi_squared = np.sum((flux - estimated_flux) ** 2 / error_squared)

    red_chi_squared = chi_squared / n_parameters_one_peak

    return red_chi_squared

def loglikelihood_one_peak(cube):

    """ 
    Calculate the logaritmic likelihood of the one-peak model for the estimated parameters

    Parameters:
        cube (list): List of (estimated) parameters of the one-peak model 

    Outputs: 
        loglikelihood (float): Logaritmic likelihood of the one-peak model
    """
      
    loglikelihood = - 0.5 * reduced_chi_squared_one_peak(cube)

    return loglikelihood

### Find best fitting parameters ###

def find_parameters_one_peak(sn_name, survey):

    """ 
    Estimate the parameters of the one-peak model using ellipsoidal nested sampling 

    Parameters:
        sn_name (str): Internal name of the SN 
        survey (str): Survey that observed the SN 

    Outputs: 
        parameter_values (numpy.ndarray): Parameters estimated by the nested sampling algorithm
        red_chi_squared (float): Mean reduced chi-squared value of the posteriors 
    """

    global time, flux, fluxerr, f1_values, f2_values

    # Folder in which sampling results will be stored
    # Not in the GitHub because some files can be too large 
    os.makedirs(f"../data/nested_sampling_results/{survey}/one_peak/{sn_name}/", exist_ok = True)
    
    # Run nested sampling to find the best parameters
    pymultinest.run(loglikelihood_one_peak, prior_one_peak, n_parameters_one_peak, n_live_points = 50,
                    outputfiles_basename = f"../data/nested_sampling_results/{survey}/one_peak/{sn_name}/",
                    resume = False, verbose = True)
    
    # Retrieve the best parameters
    analyse = pymultinest.Analyzer(n_params = n_parameters_one_peak, outputfiles_basename = f"../data/nested_sampling_results/{survey}/one_peak/{sn_name}/")
    parameter_info = analyse.get_stats()
    parameter_values = np.array(parameter_info["modes"][0]["mean"])

    # Calculate the reduced chi squared using the posteriors
    df = pd.read_csv(f"../data/nested_sampling_results/{survey}/one_peak/{sn_name}/post_equal_weights.dat", delimiter = "\t")
    posterior_samples = df.to_numpy()
    posterior_red_chi_squared = []       

    for idx in range(len(posterior_samples)):

        string = posterior_samples[idx][0]
        string_list = string.split()
        float_list = [float(num) for num in string_list]

        posterior_red_chi_squared.append(reduced_chi_squared_one_peak(float_list[:n_parameters_one_peak]))

    red_chi_squared = np.mean(posterior_red_chi_squared)
    
    return parameter_values, red_chi_squared

def run_find_parameters(sn_name, survey, return_dict):

    """ 
    Run "find_parameters_one_peak" in a seperate process in order to set a timeout

    Parameters:
        sn_name (str): Internal name of the SN 
        survey (str): Survey that observed the SN
        return_dict (dict): Dictionary in which the results of "find_parameters_one_peak" are stored

    Outputs: 
        None
    """
    
    parameters, chi_squared = find_parameters_one_peak(sn_name, survey)
    return_dict["parameter_values"] = parameters
    return_dict["red_chi_squared"] = chi_squared

def find_parameters_with_timeout(sn_name, survey, timeout = 600):

    """ 
    Run "find_parameters_one_peak" in a seperate process with a time out
    Terminate if the nested sampling algorithm takes too long  

    Parameters:
        sn_name (str): Internal name of the SN 
        survey (str): Survey that observed the SN
        timeout (int): Maximum number of seconds the algorithm can run for
        
    Outputs: 
        parameter_values (numpy.ndarray): Parameters estimated by the nested sampling algorithm
        red_chi_squared (float): Mean reduced chi-squared value of the posteriors 
    """
  
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    
    # Create a new process
    process = multiprocessing.Process(target=run_find_parameters, args=(sn_name, survey, return_dict))
    process.start()
    
    # Wait for the process to complete or timeout
    process.join(timeout)
    
    if process.is_alive():
        # If process is still running after the timeout, terminate it
        print("Nested sampling took too long. Terminating...")
        process.terminate()

        # Ensure the process has ended
        process.join()  
        
        parameter_values = np.array([])
        red_chi_squared = np.inf

    else:
        # Process completed within time limit
        parameter_values = return_dict["parameter_values"]
        red_chi_squared = return_dict["red_chi_squared"]   
        
    return parameter_values, red_chi_squared

# %%

######################################## SECOND PEAK #########################################################

### Fit second peak ###

def gaussian_distribution(time, amplitude, mean, std):
    
    """
    The Gaussian model for observations in one filter
    
    Parameters: 
        time (numpy.ndarray): Modified Julian Date of the observations in a certain filter 
        amplitude (float): Amplitude parameter of the Gaussian
        mean (float): Mean parameter of the Gaussian
        std (float): Standard deviation parameter of the Gaussian
    
    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations in a certain filter 
    """
    
    global flux, peak_main_idx
    
    estimated_flux = flux[peak_main_idx] * amplitude * np.exp(-(time - mean)**2 / (2 * std**2))
    
    return estimated_flux

def second_peak(time, amplitude, mean, std, amplitude_ratio, mean_ratio, std_ratio):
    
    """
    The Gaussian model for observations in both filters
    Used to fit the data points belonging to the second peak only
    
    Parameters: 
        time (numpy.ndarray): Modified Julian Date of the second peak observations
        amplitude (float): Amplitude parameter of the Gaussian in the f1 filter
        mean (float): Mean parameter of the Gaussian in the f1 filter
        std (float): Standard deviation parameter of the Gaussian in the f1 filter
        amplitude_ratio (float): f2/f1 parameter ratio of the amplitude of the Gaussian 
        mean_ratio (float): f2/f1 parameter ratio of the mean of the Gaussian 
        std_ratio (float): f2/f1 parameter ratio of the standard parameter of the Gaussian 
           
    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the second peak observations
    """
    
    global f1_values_peak, f2_values_peak

    estimated_flux = []

    time_f1 = time[f1_values_peak]
    estimated_flux.extend(gaussian_distribution(time_f1, amplitude, mean, std))

    time_f2 = time[f2_values_peak]
    estimated_flux.extend(gaussian_distribution(time_f2, amplitude * amplitude_ratio, mean * mean_ratio, std * std_ratio))
    
    estimated_flux = np.array(estimated_flux)
    
    return estimated_flux

def second_peak_fit(time, amplitude, mean, std, amplitude_ratio, mean_ratio, std_ratio):
    
    """
    The Gaussian model for observations in both filters
    Used to calulate the Gaussian model value for all data points belong to the SN light curve
    
    Parameters: 
        time (numpy.ndarray): Modified Julian Date of the observations
        amplitude (float): Amplitude parameter of the Gaussian in the f1 filter
        mean (float): Mean parameter of the Gaussian in the f1 filter
        std (float): Standard deviation parameter of the Gaussian in the f1 filter
        amplitude_ratio (float): f2/f1 parameter ratio of the amplitude of the Gaussian 
        mean_ratio (float): f2/f1 parameter ratio of the mean of the Gaussian 
        std_ratio (float): f2/f1 parameter ratio of the standard parameter of the Gaussian 
           
    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations
    """
   
    global f1_values, f2_values

    estimated_flux = []

    time_f1 = time[f1_values]
    estimated_flux.extend(gaussian_distribution(time_f1, amplitude, mean, std))

    time_f2 = time[f2_values]
    estimated_flux.extend(gaussian_distribution(time_f2, amplitude * amplitude_ratio, mean * mean_ratio, std * std_ratio))
    
    estimated_flux = np.array(estimated_flux)
    
    return estimated_flux

### Isolate peak ### 

def isolate_second_peak(main_filter_second_peak, extrema_f1, peak_width_f1, extrema_f2, 
                        peak_width_f2, time_aug, f1_values_aug, f2_values_aug):
   
    """
    Select the observations belonging to the second peak
    
    Parameters: 
        main_filter_second_peak (str): Filter in which the second peak is observed, if observed in both, f1 is used
        extrema_f1 (numpy.ndarray): Indices of the extrema in the f1 filter
        peak_width_f1 (float): Width of the second peak in the f1 filter
        extrema_f2 (numpy.ndarray): Indices of the extrema in the f2 filter
        peak_width_f2 (float): Width of the second peak in the f2 filter
        time_aug (numpy.ndarray): Modified Julian Date of the augmented observations 
        f1_values_aug (numpy.ndarray): Indices of augmented observations in the f1 filter
        f2_values_aug (numpy.ndarray): Indices of augmented observations in the f2 filter
           
    Outputs: 
        time_peak (numpy.ndarray): Modified Julian Date of the observations belonging to the second peak 
        flux_peak (numpy.ndarray): Differential flux of the observations belonging to the second peak in microJansky
        fluxerr_peak (numpy.ndarray): One-sigma error on the differential flux of the observations belonging to the second peak in microJansky
        filters_peak (numpy.ndarray): Used filter of observations belonging to the second peak
    """
  
    global time, flux, fluxerr, filters, f1, f2

    # Isolate the data points that are part of the peak
    if main_filter_second_peak == f1:
        peak_data = np.where(np.abs(time - time_aug[f1_values_aug][extrema_f1][2]) < peak_width_f1)

    elif main_filter_second_peak == f2:
        peak_data = np.where(np.abs(time - time_aug[f2_values_aug][extrema_f2][2]) < peak_width_f2)

    time_peak = time[peak_data]
    flux_peak = flux[peak_data]
    fluxerr_peak = fluxerr[peak_data]
    filters_peak = filters[peak_data]

    return time_peak, flux_peak, fluxerr_peak, filters_peak

# %%

######################################## FULL LIGHT CURVE ###########################################################

def calculate_flux_two_peaks(time, cube, max_flux):

    """ 
    The two-peak fitting model in one filter
    Sum of the one-peak and Gaussian models

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the observations in a certain filter 
        cube (list): List of (estimated) parameters of the fitting model in a certain filter 
        max_flux (float): Maximum flux in the f1 filter 

    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations in a certain filter 
    """
 
    # One-peak model: exponential rise in brightness
    estimated_flux = max_flux * cube[0] / (1 + np.exp(-(time - cube[1]) / cube[2]))
    
    time_constr = (time - cube[1] < cube[3])

    # One-peak model: linear plateau after peak
    estimated_flux[time_constr] *= 1 - cube[4] * (time[time_constr] - cube[1])
    
    # One-peak model: exponential decline in brightness
    estimated_flux[~time_constr] *= (1 - cube[4] * cube[3]) * np.exp((cube[3] - (time[~time_constr] - cube[1])) / cube[5])

    # Gaussian model
    estimated_flux += max_flux * cube[6] * np.exp(-(time - cube[7])**2 / (2 * cube[8]**2))

    return estimated_flux

def light_curve_two_peaks(time, cube, max_flux, f1_values, f2_values):

    """ 
    The two-peak fitting model for observations in both filters
    Sum of the one-peak and Gaussian models

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the observations 
        cube (list): List of (estimated) parameters of the fitting model 
        max_flux (float): Maximum flux in the f1 filter 
        f1_values (numpy.ndarray): Indices of observations in the f1 filter
        f2_values (numpy.ndarray): Indices of observations in the f2 filter

    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations
    """

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

    """ 
    The two-peak fitting model for observations in both filters
    Used to fit the observations using scipy's curve_fit

    Parameters:
        time (numpy.ndarray): Modified Julian Date of the observations 
        cube (list): List of (estimated) parameters of the fitting model 

    Outputs: 
        estimated_flux (numpy.ndarray): Estimated flux of the observations
    """

    global flux, f1_values, f2_values, peak_main_idx

    cube = list(cube)
    estimated_flux = light_curve_two_peaks(time, cube, flux[peak_main_idx], f1_values, f2_values)
    
    return estimated_flux

def reduced_chi_squared_two_peaks(cube):

    """ 
    Calculate the reduced chi-squared value for estimated parameters of the two-peak model

    Parameters:
        cube (list): List of (estimated) parameters of the two-peak fitting model 

    Outputs: 
        red_chi_squared (float): Reduced chi-squared value
    """
        
    global time, flux, fluxerr, f1_values, f2_values, peak_main_idx

    estimate = light_curve_two_peaks(time, cube, flux[peak_main_idx], f1_values, f2_values)

    error_param = 10 ** np.concatenate((np.full(len(f1_values[0]), cube[6]), np.full(len(f2_values[0]), cube[6] * cube[13])))
    error_squared = fluxerr ** 2 + error_param ** 2

    chi_squared = np.sum((flux - estimate) ** 2 / error_squared)

    red_chi_squared = chi_squared / n_parameters_two_peaks

    return red_chi_squared

# %% 

########################################## PLOTTING ####################################################


def plot_data_augmentation_extrema(sn_name, time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug, peak_time, extrema_f1, extrema_f2, save_name):

    """
    Plot the augmented data and its extrema 

    Parameters: 
        sn_name (str): Internal name of the SN
        time_aug (numpy.ndarray): Modified Julian Date of the augmented observations 
        flux_aug (numpy.ndarray): Differential flux of the augmented observations in microJansky
        fluxerr_aug (numpy.ndarray): One-sigma error on the differential flux of the augmented observations in microJansky
        f1_values_aug (numpy.ndarray): Indices of augmented observations in the f1 filter
        f2_values_aug (numpy.ndarray): Indices of augmented observations in the f2 filter
        peak_time (float): Modified Julian Date of the main peak in the f1 filter
        extrema_f1 (numpy.ndarray): Indices of the extrema in the f1 filter
        extrema_f2 (numpy.ndarray): Indices of the extrema in the f2 filter
        save_name (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs: 
        None
    """
    
    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    # the extrema
    plt.scatter(time_aug[f1_values_aug][extrema_f1] + peak_time, flux_aug[f1_values_aug][extrema_f1], s = 250, marker = "*", c = colours["green"], edgecolors = "black", label = "Extrema r-band" , zorder = 10)

    # the original and augmented data
    plt.fill_between(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug] - fluxerr_aug[f1_values_aug], flux_aug[f1_values_aug] + fluxerr_aug[f1_values_aug], color = colours["blue"], alpha = 0.1)
    plt.errorbar(time[f1_values] + peak_time, flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = colours["blue"], label = "Band: r", zorder = 5)
    plt.errorbar(time_aug[f1_values_aug] + peak_time, flux_aug[f1_values_aug], yerr = fluxerr_aug[f1_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = colours["blue"], zorder = 5)

    # the extrema
    plt.scatter(time_aug[f2_values_aug][extrema_f2] + peak_time, flux_aug[f2_values_aug][extrema_f2], s = 250, marker = "*", c = colours["red"], edgecolors = "black", label = "Extrema g-band", zorder = 10)

    # the original and augmented data
    plt.fill_between(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug] - fluxerr_aug[f2_values_aug], flux_aug[f2_values_aug] + fluxerr_aug[f2_values_aug], color = colours["orange"], alpha = 0.1)
    plt.errorbar(time[f2_values] + peak_time, flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, alpha = 0.9, color = colours["orange"], label = "Band: g", zorder = 5)
    plt.errorbar(time_aug[f2_values_aug] + peak_time, flux_aug[f2_values_aug], yerr = fluxerr_aug[f2_values_aug], fmt = "o", markersize = 4, capsize = 2, alpha = 0.1, color = colours["orange"], zorder = 5)

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {sn_name}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    
    if not save_name:
        plt.show()

    else: 
        plt.savefig(f"../plots/data_augmentation_extrema/{save_name}", dpi = 300, bbox_inches = "tight")
        plt.close()
    
def plot_best_fit_light_curve(sn_name, red_chi_squared, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, save_name = False):

    """
    Plot the best-fit light curve 

    Parameters: 
        sn_name (str): Internal name of the SN
        red_chi_squared (float): Reduced chi-squared value
        time_fit (numpy.ndarray): Modified Julian Date of the fit data points 
        flux_fit (numpy.ndarray): Differential flux of the fit data points in microJansky
        f1_values_fit (numpy.ndarray): Indices of fit data points in the f1 filter
        f2_values_fit (numpy.ndarray): Indices of fit data points in the f2 filter
        peak_time (float): Modified Julian Date of the main peak in the f1 filter
        save_name (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs: 
        None
    """
  
    global time, flux, fluxerr, filters, f1_values, f2_values, f1, f2

    plt.plot(time_fit[f1_values_fit] + peak_time, flux_fit[f1_values_fit], linestyle = "--", linewidth = 2, color = colours["blue"], label = f"Best-fit light curve {f1}-band")
    plt.errorbar(time[f1_values] + peak_time, flux[f1_values], yerr = fluxerr[f1_values], fmt = "o", markersize = 4, capsize = 2, color = colours["blue"], label = f"Band: {f1}", zorder = 5)

    plt.plot(time_fit[f2_values_fit] + peak_time, flux_fit[f2_values_fit], linestyle = "--", linewidth = 2, color = colours["orange"], label = f"Best-fit light curve {f2}-band")                
    plt.errorbar(time[f2_values] + peak_time, flux[f2_values], yerr = fluxerr[f2_values], fmt = "o", markersize = 4, capsize = 2, color = colours["orange"], label = f"Band: {f2}", zorder = 5)

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {sn_name}. " + r"$\mathrm{X}^{2}_{red}$" + f" = {red_chi_squared:.2f}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    
    if not save_name:
        plt.show()

    else: 
        plt.savefig(f"../plots/best-fit/{save_name}", dpi = 300, bbox_inches = "tight")
        plt.close()

def fit_light_curve(sn_name, survey):

    """
    Fit a SN light curve 
    
    Parameters: 
        sn_name (str): Internal name of the SN 
        survey (str): Survey that observed the SN

    Outputs: 
        None
    """
    
    print(sn_name)

    global parameter_bounds_gaussian, parameter_bounds_two_peaks
    global f1, f2
    global time, flux, fluxerr, filters, f1_values, f2_values, peak_main_idx, f1_values_peak, f2_values_peak

    ########################################## PRE-PROCESSING ####################################################

    ###### Retrieve data ######
    if survey == "ZTF":
        f1 = "r"
        f2 = "g"

        # Load the data
        time, flux, fluxerr, filters = ztf_load_data(sn_name)

    if survey == "ATLAS":
        f1 = "o"
        f2 = "c"

        # Load the data
        time, flux, fluxerr, filters = atlas_load_data(sn_name)

    f1_values = np.where(filters == f1)
    f2_values = np.where(filters == f2)

    # Shift the light curve so that the main peak is at 0 MJD
    peak_main_idx = np.argmax(flux[f1_values])
    peak_time = np.copy(time[peak_main_idx])
    peak_flux = np.copy(flux[peak_main_idx])

    time -= peak_time

    ###### Augment data ######
    amount_aug = 65

    time_aug, flux_aug, fluxerr_aug, filters_aug = augment_data(survey, amount_aug)

    f1_values_aug = np.where(filters_aug == f1)
    f2_values_aug = np.where(filters_aug == f2)

    ###### Find extrema ######
    extrema_f1, peak_width_f1, extrema_f2, peak_width_f2 = find_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug)

    ########################################## LIGHT CURVE FITTING ####################################################

    # Data for plotting
    amount_fit = 100

    # Plot from 30 days before the first detection to 30 days after the last detection
    time_fit = np.concatenate((np.linspace(time.min() - 30, time.max() + 30, amount_fit), np.linspace(time.min() - 30, time.max() + 30, amount_fit)))
    f1_values_fit = np.arange(amount_fit)
    f2_values_fit = np.arange(amount_fit) + amount_fit

    # No additional peaks were detected in both filters 
    if extrema_f1[2] == -1 and extrema_f2[2] == -1:
        # Only a one-peak light curve can be fit through the data
        one_peak_parameters, red_chi_squared = find_parameters_with_timeout(sn_name, survey)

        # Nested samplign successfully estimated the parameter values
        if len(one_peak_parameters) != 0:
            np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)
            
            red_chi_squared = reduced_chi_squared_one_peak(one_peak_parameters)
            with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow([sn_name, red_chi_squared])
            
            flux_fit = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
            plot_best_fit_light_curve(sn_name, red_chi_squared, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

        # Nested sampling took too long and the process was cancelled
        else: 
            # Remove SN from sample because it cannot be fit
            if sn_name in np.loadtxt(f"../data/processed/{survey}/{survey}_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str"):
                file_name = f"../data/processed/{survey}/{survey}_SNe_Ia_CSM.txt"

            elif sn_name in np.loadtxt(f"../data/processed/{survey}/{survey}_SNe_IIn.txt", delimiter = ",", dtype = "str"):
                file_name = f"../data/processed/{survey}/{survey}_SNe_IIn.txt"
                
            with open(file_name, "r") as file:             
                SN_names = file.readlines()

            with open(file_name, "w") as file:
                for name in SN_names:

                    if name.strip("\n") != sn_name:
                        file.write(name)

     # There is a possibility that there is a second peak
    else: 
        ### Fit a one-peak light curve
        one_peak_parameters, red_chi_squared_OP = find_parameters_with_timeout(sn_name, survey)

        # Nested sampling took too long and the process was cancelled
        if len(one_peak_parameters) == 0:
            # Remove light curve from list because it cannot be fit
            if sn_name in np.loadtxt(f"../data/processed/{survey}/{survey}_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str"):
                file_name = f"../data/processed/{survey}/{survey}_SNe_Ia_CSM.txt"

            elif sn_name in np.loadtxt(f"../data/processed/{survey}/{survey}_SNe_IIn.txt", delimiter = ",", dtype = "str"):
                file_name = f"../data/processed/{survey}/{survey}_SNe_IIn.txt"
                
            with open(file_name, "r") as file:             
                SN_names = file.readlines()

            with open(f"../data/processed/{survey}/{survey}_SNe_Ia_CSM.txt", "w") as file:
                for name in SN_names:

                    if name.strip("\n") != sn_name:
                        file.write(name)

            return
            
        ## Fit a two-peaks light curve
        # Identify the main filter for the second peak (f1 if both have peaks)
        if extrema_f1[2] != -1:
            main_filter_second_peak = f1

        elif extrema_f2[2] != -1:
            main_filter_second_peak = f2

        time_peak, flux_peak, fluxerr_peak, filters_peak = isolate_second_peak(main_filter_second_peak, extrema_f1, peak_width_f1, 
                                                                               extrema_f2, peak_width_f2, time_aug, f1_values_aug, f2_values_aug)

        # There must be more fitting data points than fitting parameters (= 6)
        if len(time_peak) < 6:
            red_chi_squared_TP = np.inf

        else:
            f1_values_peak = np.where(filters_peak == f1)
            f2_values_peak = np.where(filters_peak == f2)

            # Inital guess of the parameters (amplitude, mean, standard deviation)
            # If in both filters a second peak is detected
            if extrema_f1[2] != -1 and extrema_f2[2] != -1:
                guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 
                                            flux_aug[f2_values_aug][extrema_f2][2] / flux_aug[f1_values_aug][extrema_f1][2], 
                                            time_aug[f2_values_aug][extrema_f2][2] / time_aug[f1_values_aug][extrema_f1][2], 
                                            peak_width_f2 / peak_width_f1])
            
            # If only in one peak a second peak is detected
            else:
                if extrema_f1[2] != -1:
                    guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 1, 1, 1])

                elif extrema_f2[2] != -1:
                    guess_parameters = np.array([1, time_aug[f2_values_aug][extrema_f2][2], peak_width_f2, 1, 1, 1])

            down_bound = parameter_bounds_gaussian[0]
            up_bound = parameter_bounds_gaussian[1]

            # The inital guess must be within these ranges to avoid unrealistic peaks
            if not ((down_bound <= list(guess_parameters)) & (list(guess_parameters) <= up_bound)):

                # The one-peak light curve is a better fit
                np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)
                
                with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                    writer = csv.writer(file)
                    writer.writerow([sn_name, red_chi_squared_OP])

                # Plot the results
                flux_fit = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
                plot_best_fit_light_curve(sn_name, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

                return 
            
            else:

                # Estimate the parameters of the second peak in both bands simultaneously
                try:
                    second_peak_parameters, _ = curve_fit(second_peak, time_peak, flux_peak, p0 = guess_parameters, sigma = fluxerr_peak, bounds = parameter_bounds_gaussian)
                
                except (RuntimeError, ValueError):        
                    # The one-peak light curve is a better fit
                    np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)

                    with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                        writer = csv.writer(file)
                        writer.writerow([sn_name, red_chi_squared_OP])

                    # Plot the results
                    flux_fit = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
                    plot_best_fit_light_curve(sn_name, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

                    return

                # Calculate the predicted flux of the second peak
                flux_peak_fit = second_peak_fit(time, *second_peak_parameters)

                # Calculate the flux without the second peak
                original_flux = np.copy(flux)
                flux -= flux_peak_fit

                ### Fit the residual light curve
                residual_parameters, _ = find_parameters_with_timeout(sn_name, survey)

                # Nested sampling took too long and the process was cancelled
                if len(residual_parameters) == 0:
                    # The one-peak light curve is a better fit
                    np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)

                    with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                        writer = csv.writer(file)
                        writer.writerow([sn_name, red_chi_squared_OP])

                    # Plot the results
                    flux_fit = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
                    plot_best_fit_light_curve(sn_name, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

                    return

                ### Fit the two-peak model using the fit parameters of the seperate models
                flux = original_flux

                initial_guesses = list(residual_parameters) + list(second_peak_parameters)

                # Estimate the parameters of the full light curve in both bands simultaneously
                try:
                    two_peaks_parameters, _ = curve_fit(function_approximation_two_peaks, time, flux, p0 = initial_guesses, sigma = fluxerr, bounds = parameter_bounds_two_peaks)
                
                except (RuntimeError, ValueError):        
                    # The one-peak light curve is a better fit
                    np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)

                    with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                        writer = csv.writer(file)
                        writer.writerow([sn_name, red_chi_squared_OP])

                    # Plot the results
                    flux_fit = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
                    plot_best_fit_light_curve(sn_name, red_chi_squared_OP, time_fit, flux_fit, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

                    return

                red_chi_squared_TP = reduced_chi_squared_two_peaks(two_peaks_parameters)

        ### Based on best red chi squared value save one- or two-peak result
        if np.abs(red_chi_squared_OP - 1) < np.abs(red_chi_squared_TP - 1):
            # The one-peak light curve is a better fit
            np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)
            
            with open(f"../data/best-fit/{survey}/one_peak/red_chi_squared_OP.csv", "a", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow([sn_name, red_chi_squared_OP])

            # Plot the results
            flux_fit_OP = light_curve_one_peak(time_fit, one_peak_parameters, peak_flux, f1_values_fit, f2_values_fit)
            plot_best_fit_light_curve(sn_name, red_chi_squared_OP, time_fit, flux_fit_OP, f1_values_fit, f2_values_fit, peak_time, f"{survey}/one_peak/Best_fit_{sn_name}_OP")

        else:
            # The two-peaks light curve is a better fit
            np.save(f"../data/best-fit/{survey}/two_peaks/{sn_name}_parameters_TP", two_peaks_parameters) 

            # Still save one-peak parameters for further analysis
            np.save(f"../data/best-fit/{survey}/one_peak/{sn_name}_parameters_OP", one_peak_parameters)

            with open(f"../data/best-fit/{survey}/two_peaks/red_chi_squared_TP.csv", "a", newline = "") as file:
                writer = csv.writer(file)
                writer.writerow([sn_name, red_chi_squared_TP])

            # Plot the results
            flux_fit_TP = light_curve_two_peaks(time_fit, two_peaks_parameters, peak_flux, f1_values_fit, f2_values_fit)
            plot_best_fit_light_curve(sn_name, red_chi_squared_TP, time_fit, flux_fit_TP, f1_values_fit, f2_values_fit, peak_time, f"{survey}/two_peaks/Best_fit_{sn_name}_TP")

# %%

if __name__ == '__main__':

    # ZTF
    survey = "ZTF"

    # Load SNe names
    ztf_names_sn_Ia_CSM = np.loadtxt("../data/processed/ZTF/ZTF_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str")
    ztf_names_sn_IIn = np.loadtxt("../data/processed/ZTF/ZTF_SNe_IIn.txt", delimiter = ",", dtype = "str")
    ztf_names = np.concatenate((ztf_names_sn_Ia_CSM, ztf_names_sn_IIn))
    
    for sn_name in ztf_names:
        
        fit_light_curve(sn_name, survey)

    # ATLAS
    survey = "ATLAS"

    # Load SNe names
    atlas_names_sn_Ia_CSM = np.loadtxt("../data/processed/ATLAS/ATLAS_SNe_Ia_CSM.txt", delimiter = ",", dtype = "str")
    atlas_names_sn_IIn = np.loadtxt("../data/processed/ATLAS/ATLAS_SNe_IIn.txt", delimiter = ",", dtype = "str")
    atlas_names = np.concatenate((atlas_names_sn_Ia_CSM, atlas_names_sn_IIn))

    for sn_name in atlas_names:

        fit_light_curve(sn_name, survey)

# %%
