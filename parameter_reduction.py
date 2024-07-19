# %%

from data_processing import get_by_ztf_object_id, load_ztf_data, retrieve_atlas_data, plot_ztf_data, data_augmentation, plot_data_augmentation

# from sklearn.preprocessing import LabelEncoder
# import seaborn as sn
# from sklearn import decomposition, metrics
# from sklearn.manifold import TSNE
# import umap.umap_ as umap
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import truncnorm, uniform, norm
import pandas as pd
import json
import os

# %%

ztf_id_sn_Ia_CSM = np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn = np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")
        
ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))
ztf_types = np.concatenate((np.zeros(len(ztf_id_sn_Ia_CSM)), np.ones(len(ztf_id_sn_IIn))))
ztf_types_name = np.array(["Ia_CSM" if ztf_types[idx] == 0 else "IIn" for idx in range(len(ztf_types))])

# %%
### Analytical light curve fitting

# Using PyMultiNest

global mean, std
global time, flux, fluxerr, filters, r_values, g_values, peak_main_idx

parameters = ["A_r", "t_0_r", "t_rise_r", "gamma_r", "beta_r", "t_fall_r", "error_r", \
              "A_g", "t_0_g", "t_gise_g", "gamma_g", "beta_g", "t_fall_g", "error_g", \
              "A_r", "t_0_r", "t_rise_r", "gamma_r", "beta_r", "t_fall_r", "error_r", \
              "A_g", "t_0_g", "t_gise_g", "gamma_g", "beta_g", "t_fall_g", "error_g",]

n_params = len(parameters)

std_scale = 2
mean = [0.1457, -7.878, 0.6664, 1.4258, 0.00833, 1.5261, -1.6629,  -0.0766, -3.4065, -0.1510, -0.0452, -0.2068, -0.1486, -0.1509] \
     + [-1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]
std = [std_scale * 0.0575, std_scale * 9.916, std_scale * 0.4250, std_scale * 0.5079, std_scale * 0.00585, std_scale * 0.5057, std_scale * 0.5578, \
       std_scale * 0.1096, std_scale * 4.566, std_scale * 0.1927, std_scale * 0.1705, std_scale * 0.2704, std_scale * 0.2642, std_scale * 0.2542] + [0.05] * int(n_params/2)

def prior_one_peak(cube, ndim, nparams):

    ### First peak

    # A 1
    min = (-0.3 - mean[0]) / std[0]
    max = (0.5 - mean[0]) / std[0]
    cube[0] = truncnorm.ppf(cube[0], min, max, mean[0], std[0])

    # t_0 1
    min = (-100.0 -mean[1]) / std[1]
    max = (30.0 - mean[1]) / std[1]
    cube[1] = truncnorm.ppf(cube[1], min, max, mean[1], std[1])

    # t_rise 1
    min = (-2.0 - mean[2]) / std[2]
    max = (4.0 - mean[2]) / std[2]
    cube[2] = truncnorm.ppf(cube[2], min, max, mean[2], std[2])

    # gamma 1
    min = (0.0 - mean[3]) / std[3]
    max = (3.5 - mean[3]) / std[3]
    cube[3] = truncnorm.ppf(cube[3], min, max, mean[3], std[3])

    # beta 1
    min = (0.0 - mean[4]) / std[4]
    max = (0.03 - mean[4]) / std[4]
    cube[4] = truncnorm.ppf(cube[4], min, max, mean[4], std[4])

    # t_fall 1
    min = (0.0 - mean[5]) / std[5]
    max = (4.0 - mean[5]) / std[5]
    cube[5] = truncnorm.ppf(cube[5], min, max, mean[5], std[5])

    # error 1
    min = (-3.0 - mean[6]) / std[6]
    max = (-0.8 - mean[6]) / std[6]
    cube[6] = truncnorm.ppf(cube[6], min, max, mean[6], std[6])

    # A 2
    min = (-1.0 - mean[7]) / std[7]
    max = (1.0 - mean[7]) / std[7]
    cube[7] = truncnorm.ppf(cube[7], min, max, mean[7], std[7])

    # t_0 2
    min = (-50.0 - mean[8]) /std[8] 
    max = (30.0 - mean[8]) / std[8]
    cube[8] = truncnorm.ppf(cube[8], min, max, mean[8], std[8])

    # t_rise 2
    min = (-1.5 - mean[9]) / std[9]
    max = (1.5 - mean[9]) / std[9]
    cube[9] = truncnorm.ppf(cube[9], min, max, mean[9], std[9])

    # gamma 2
    min = (-1.5 - mean[10]) / std[10]
    max = (1.5 - mean[10]) / std[10]
    cube[10] = truncnorm.ppf(cube[10], min, max, mean[10], std[10])

    # beta 2
    min = (-2.0 - mean[11]) / std[11]
    max = (1.0 - mean[11]) / std[11]
    cube[11] = truncnorm.ppf(cube[11], min, max, mean[11], std[11])

    # t_fall 2
    min = (-1.5 - mean[12]) / std[12]
    max = (1.5 - mean[12]) / std[12]
    cube[12] = truncnorm.ppf(cube[12], min, max, mean[12], std[12])

    # error 2
    min = (-1.5 - mean[13]) / std[13]
    max = (-1.0 - mean[13]) / std[13]
    cube[13] = truncnorm.ppf(cube[13], min, max, mean[13], std[13])

    ### Second peak

    # A 1
    cube[14] = uniform.ppf(cube[14], -1, 0.05)

    # t_0 1
    cube[15] = uniform.ppf(cube[15], 0, 0.05)

    # t_rise 1
    cube[16] = uniform.ppf(cube[16], 0, 0.05)

    # gamma 1
    cube[17] = uniform.ppf(cube[17], 0, 0.05)

    # beta 1
    cube[18] = uniform.ppf(cube[18], 1, 0.05)

    # t_fall 1
    cube[19] = uniform.ppf(cube[19], 0, 0.05)

    # error 1
    cube[20] = uniform.ppf(cube[20], 0, 0.05)

    # A 2
    cube[21] = uniform.ppf(cube[21], 1, 0.05)

    # t_0 2
    cube[22] = uniform.ppf(cube[22], 1, 0.05)

    # t_rise 2
    cube[23] = uniform.ppf(cube[23], 1, 0.05)

    # gamma 2
    cube[24] = uniform.ppf(cube[24], 1, 0.05)

    # beta 2
    cube[25] = uniform.ppf(cube[25], 1, 0.05)

    # t_fall 2
    cube[26] = uniform.ppf(cube[26], 1, 0.05)

    # error 2
    cube[27] = uniform.ppf(cube[27], 1, 0.05)

    return cube

def prior_double_peak(cube, ndim, nparams):

    ### First peak

    # A 1
    mean = 0.1457
    std = 2 * 0.0575

    min = (-0.3 - mean) / std
    max = (0.5 - mean) / std
    cube[0] = truncnorm.ppf(cube[0], min, max, mean, std)
    # cube[0] = 0.7 * cube[0] - 0.3

    # t_0 1
    mean = -7.878
    std = 2 * 9.916

    min = (-100.0 - mean) / std
    max = (30.0 - mean) / std 
    cube[1] = truncnorm.ppf(cube[1], min, max, mean, std)
    # cube[1] = 130 * cube[1] - 100

    # t_rise 1
    mean = 0.6664
    std = 2 * 0.4250

    min = (-2.0 - mean) / std
    max = (4.0 - mean) / std 
    cube[2] = truncnorm.ppf(cube[2], min, max, mean, std)
    # cube[2] = 6.0 * cube[2] - 2.0

    # gamma 1
    mean = 1.4258
    std = 2 * 0.3079

    min = (0.0 - mean) / std
    max = (3.5 - mean) / std 
    cube[3] = truncnorm.ppf(cube[3], min, max, mean, std)
    # cube[3] = 3.5 * cube[3]

    # beta 1
    mean = 0.00833
    std = 2 * 0.00385

    min = (0.0 - mean) / std
    max = (0.03 - mean) / std 
    cube[4] = truncnorm.ppf(cube[4], min, max, mean, std)
    # cube[4] = 0.03 * cube[4]

    # t_fall 1
    mean = 1.5261
    std = 2 * 0.3037

    min = (0.0 - mean) / std
    max = (4.0 - mean) / std 
    cube[5] = truncnorm.ppf(cube[5], min, max, mean, std)
    # cube[5] = 4.0 * cube[5]

    # error 1
    mean = -1.6629
    std = 2 * 0.3378

    min = (-3.0 - mean) / std
    max = (-0.8 - mean) / std 
    cube[6] = truncnorm.ppf(cube[6], min, max, mean, std)
    # cube[6] = 2.2 * cube[6] - 3.0

    # A 2
    mean = -0.0766
    std = 2 * 0.1096

    min = (-1.0 - mean) / std
    max = (1.0 - mean) / std
    cube[7] = truncnorm.ppf(cube[7], min, max, mean, std)
    # cube[7] = 2.0 * cube[7] - 1.0

    # t_0 2
    mean = -3.4065
    std = 2 * 4.366

    min = (-50.0 - mean) / std
    max = (30.0 - mean) / std 
    cube[8] = truncnorm.ppf(cube[8], min, max, mean, std)
    # cube[8] = 80.0 * cube[8] - 50.0

    # t_rise 2
    mean = -0.1510
    std = 2 * 0.1927

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[9] = truncnorm.ppf(cube[9], min, max, mean, std)
    # cube[9] = 3.0 * cube[9] - 1.5

    # gamma 2
    mean = -0.0452
    std = 2 * 0.1705

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[10] = truncnorm.ppf(cube[10], min, max, mean, std)
    # cube[10] = 3.0 * cube[10] - 1.5

    # beta 2
    mean = -0.2068
    std = 2 * 0.2704

    min = (-2.0 - mean) / std
    max = (1.0 - mean) / std 
    cube[11] = truncnorm.ppf(cube[11], min, max, mean, std)
    # cube[11] = 3.0 * cube[11] - 2.0

    # t_fall 2
    mean = -0.1486
    std = 2 * 0.2642

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[12] = truncnorm.ppf(cube[12], min, max, mean, std)
    # cube[12] = 3.0 * cube[12] - 1.5

    # error 2
    mean = -0.1509
    std = 2 * 0.2542

    min = (-1.5 - mean) / std
    max = (-1.0 - mean) / std 
    cube[13] = truncnorm.ppf(cube[13], min, max, mean, std)
    # cube[13] = 0.5 * cube[13] - 1.5

    ### Second peak

    # A 1
    mean = 0.1457
    std = 2 * 0.0575

    min = (-0.3 - mean) / std
    max = (0.5 - mean) / std
    cube[14] = truncnorm.ppf(cube[14], min, max, mean, std)
    # cube[0] = 0.7 * cube[0] - 0.3

    # t_0 1
    mean = -7.878
    std = 2 * 9.916

    min = (-100.0 - mean) / std
    max = (30.0 - mean) / std 
    cube[15] = truncnorm.ppf(cube[15], min, max, mean, std)
    # cube[1] = 130 * cube[1] - 100

    # t_rise 1
    mean = 0.6664
    std = 2 * 0.4250

    min = (-2.0 - mean) / std
    max = (4.0 - mean) / std 
    cube[16] = truncnorm.ppf(cube[16], min, max, mean, std)
    # cube[2] = 6.0 * cube[2] - 2.0

    # gamma 1
    mean = 1.4258
    std = 2 * 0.3079

    min = (0.0 - mean) / std
    max = (3.5 - mean) / std 
    cube[17] = truncnorm.ppf(cube[17], min, max, mean, std)
    # cube[3] = 3.5 * cube[3]

    # beta 1
    mean = 0.00833
    std = 2 * 0.00385

    min = (0.0 - mean) / std
    max = (0.03 - mean) / std 
    cube[18] = truncnorm.ppf(cube[18], min, max, mean, std)
    # cube[4] = 0.03 * cube[4]

    # t_fall 1
    mean = 1.5261
    std = 2 * 0.3037

    min = (0.0 - mean) / std
    max = (4.0 - mean) / std 
    cube[19] = truncnorm.ppf(cube[19], min, max, mean, std)
    # cube[5] = 4.0 * cube[5]

    # error 1
    mean = -1.6629
    std = 2 * 0.3378

    min = (-3.0 - mean) / std
    max = (-0.8 - mean) / std 
    cube[20] = truncnorm.ppf(cube[20], min, max, mean, std)
    # cube[6] = 2.2 * cube[6] - 3.0

    # A 2
    mean = -0.0766
    std = 2 * 0.1096

    min = (-1.0 - mean) / std
    max = (1.0 - mean) / std
    cube[21] = truncnorm.ppf(cube[21], min, max, mean, std)
    # cube[7] = 2.0 * cube[7] - 1.0

    # t_0 2
    mean = -3.4065
    std = 2 * 4.366

    min = (-50.0 - mean) / std
    max = (30.0 - mean) / std 
    cube[22] = truncnorm.ppf(cube[22], min, max, mean, std)
    # cube[8] = 80.0 * cube[8] - 50.0

    # t_rise 2
    mean = -0.1510
    std = 2 * 0.1927

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[23] = truncnorm.ppf(cube[23], min, max, mean, std)
    # cube[9] = 3.0 * cube[9] - 1.5

    # gamma 2
    mean = -0.0452
    std = 2 * 0.1705

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[24] = truncnorm.ppf(cube[24], min, max, mean, std)
    # cube[10] = 3.0 * cube[10] - 1.5

    # beta 2
    mean = -0.2068
    std = 2 * 0.2704

    min = (-2.0 - mean) / std
    max = (1.0 - mean) / std 
    cube[25] = truncnorm.ppf(cube[25], min, max, mean, std)
    # cube[11] = 3.0 * cube[11] - 2.0

    # t_fall 2
    mean = -0.1486
    std = 2 * 0.2642

    min = (-1.5 - mean) / std
    max = (1.5 - mean) / std 
    cube[26] = truncnorm.ppf(cube[26], min, max, mean, std)
    # cube[12] = 3.0 * cube[12] - 1.5

    # error 2
    mean = -0.1509
    std = 2 * 0.2542

    min = (-1.5 - mean) / std
    max = (-1.0 - mean) / std 
    cube[27] = truncnorm.ppf(cube[27], min, max, mean, std)
    # cube[13] = 0.5 * cube[13] - 1.5

    return cube

def calculate_flux(time, cube, max_flux):

    estimated_flux = max_flux * cube[0] / (1 + np.exp(-(time - cube[1]) / cube[2]))
    
    time_constr = (time - cube[1] < cube[3])

    estimated_flux[time_constr] *= 1 - cube[4] * (time[time_constr] - cube[1])
    estimated_flux[~time_constr] *= (1 - cube[4] * cube[3]) * np.exp((cube[3] - (time[~time_constr] - cube[1])) / cube[5])

    return estimated_flux

def light_curve_approximation(time, max_flux, data_filter_1, data_filter_2, cube):

    estimated_flux = []

    time_filter_1 = time[data_filter_1]
    cube_filter_1 = [10 ** cube[0], cube[1], 10 ** cube[2], 10 ** cube[3], cube[4], 10 **cube[5]]
    estimated_flux.extend(calculate_flux(time_filter_1, cube_filter_1, max_flux))

    time_filter_2 = time[data_filter_2]
    cube_filter_2 = [10 ** (cube[0] + cube[7]), cube[1] + cube[8], 10 ** (cube[2] + cube[9]), 10 ** (cube[3] + cube[10]), \
                     cube[4] * (10 ** cube[11]), 10 ** (cube[5] + cube[12])]
    estimated_flux.extend(calculate_flux(time_filter_2, cube_filter_2, max_flux))

    return np.array(estimated_flux)

def reduced_chi_squared(cube):

        estimate_1 = light_curve_approximation(time, flux[peak_main_idx], r_values, g_values, cube[0:14])
        estimate_2 = light_curve_approximation(time, flux[peak_main_idx], r_values, g_values, cube[14:28])
        estimate = estimate_1 + estimate_2

        error_param_1 = 10 ** np.concatenate((np.full(len(r_values[0]), cube[6]), np.full(len(g_values[0]), cube[6] * cube[13])))
        error_param_2 = 10 ** np.concatenate((np.full(len(r_values[0]), cube[20]), np.full(len(g_values[0]), cube[20] * cube[27])))
        error_squared = fluxerr ** 2 + error_param_1 ** 2 + error_param_2 ** 2

        chi_squared = np.sum((flux - estimate) ** 2 / error_squared)

        return chi_squared / n_params

def loglikelihood(cube, ndim, nparams):

    return - 0.5 * reduced_chi_squared(cube)

def plot_best_fit_light_curve(SN_id, red_chi_squared, filter_1, filter_2, time, flux, fluxerr, filter_1_values, 
                              filter_2_values, time_fit, flux_fit, save_fig = False):

    if len(filter_1_values[0]) != 0:
        plt.errorbar(time[filter_1_values], flux[filter_1_values], yerr = fluxerr[filter_1_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = f"Samples {filter_1}-band")
    
    if len(filter_2_values[0]) != 0:
        plt.errorbar(time[filter_2_values], flux[filter_2_values], yerr = fluxerr[filter_2_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = f"Samples {filter_2}-band")

    plt.plot(time_fit[:1000], flux_fit[:1000], linestyle = "--", linewidth = 1, alpha = 0.5, color = "tab:blue", label = f"Best-fitted light curve {filter_1}-band")
    plt.plot(time_fit[1000:], flux_fit[1000:], linestyle = "--", linewidth = 1, alpha = 0.5, color = "tab:orange", label = f"Best-fitted light curve {filter_2}-band")
    
    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}. " + r"$\mathrm{X}^{2}_{red}$" + f" = {red_chi_squared:.2f}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"ZTF_analytical_lightcurves_plots/LC_{SN_id}", dpi = 300)
        plt.close()
    else:
        plt.show()

for SN_id in ["ZTF18acvgjqv"]: #ztf_id[:1]:
        
    time, flux, fluxerr, filters = load_ztf_data(SN_id)

    r_values = np.where(filters == "r")
    g_values = np.where(filters == "g")

    if "r" in filters:
        peak_main_idx = np.argmax(flux[r_values])

    # There is only data in the g-band
    # The g-band becomes the main band 
    else: 
        peak_main_idx = np.argmax(flux[g_values])

    time -= time[peak_main_idx]

    # estimated_parameters = []
    # estimated_parameters_uncertainties = []

    done = True
    red_chi_squared = reduced_chi_squared(mean)
    print("Reduced chi-square:", red_chi_squared)

    time_data = np.copy(time)
    flux_data = np.copy(flux)
    fluxerr_data = np.copy(fluxerr)
    filters_data = np.copy(filters)

    passbands, passband2lam, augmentation = data_augmentation("ZTF", time_data, flux_data,
                                                              fluxerr_data, filters, "GP")

    time, flux, fluxerr, filters = augmentation.augmentation(time_data.min() + time_data[peak_main_idx], time_data.max() + time_data[peak_main_idx], n_obs = 10)

    time = np.append(time, time_data)
    flux = np.append(flux, flux_data)
    fluxerr = np.append(fluxerr, fluxerr_data)
    filters = np.append(filters, filters_data)
    
    r_values = np.where(filters == "r")
    g_values = np.where(filters == "g")

    if "r" in filters:
        peak_main_idx = np.argmax(flux[r_values])

    else: 
        peak_main_idx = np.argmax(flux[g_values])

    time -= time[peak_main_idx]

    fig, ax = plt.subplots()
    plot_data_augmentation(SN_id, passbands, passband2lam, "GP", time_data, flux_data, fluxerr_data, 
                           time, flux, fluxerr, filters, ax)
    plt.show()
    plt.close(fig)
    
    # time_fit = np.concatenate((np.linspace(time_data.min(), time_data.max(), 1000), np.linspace(time_data.min(), time_data.max(), 1000))) + time_data[peak_main_idx]
    time_fit = np.concatenate((np.linspace(time.min(), time.max(), 1000), np.linspace(time.min(), time.max(), 1000))) + time[peak_main_idx]
    print(time_fit.min(), time_fit.max())
    r_values_fit = np.arange(1000)
    g_values_fit = np.arange(1000) + 1000
    flux_fit = light_curve_approximation(time_fit - time[peak_main_idx], flux[peak_main_idx], r_values_fit, g_values_fit, mean)
    
    # plot_best_fit_light_curve(SN_id, red_chi_squared, "r", "g", time, flux, fluxerr,  
    #                                   r_values, g_values, time_fit, flux_fit, save_fig = False)
    
    plt.plot(time_fit[:1000], flux_fit[:1000], linestyle = "--", linewidth = 1, alpha = 0.5, color = "tab:blue", label = f"Best-fitted light curve -band")
    plt.plot(time_fit[1000:], flux_fit[1000:], linestyle = "--", linewidth = 1, alpha = 0.5, color = "tab:orange", label = f"Best-fitted light curve -band")
    
    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {SN_id}. " + r"$\mathrm{X}^{2}_{red}$" + f" = {red_chi_squared:.2f}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

    # while done:

    #     for amount, prior in enumerate([prior_one_peak]): #, prior_double_peak

    #         os.makedirs(f"ZTF_analytical_params/{SN_id}/{amount}_peak/", exist_ok = True)
            
    #         pymultinest.run(loglikelihood, prior, n_params, 
    #                     outputfiles_basename = f"ZTF_analytical_params/{SN_id}/{amount}_peak/",
    #                     resume = False, verbose = True)
            
    #         analyse = pymultinest.Analyzer(n_params = n_params, outputfiles_basename = f"ZTF_analytical_params/{SN_id}/{amount}_peak/")
    #         parameter_info = analyse.get_stats()
    #         parameter_values = parameter_info["modes"][0]["mean"]
    #         parameter_sigma = parameter_info["modes"][0]["sigma"]
    #         # estimated_parameters.append(parameter_values)
    #         # estimated_parameters_uncertainties.append(parameter_sigma)

    #         flux_fit = light_curve_approximation(time_fit - time[peak_main_idx], flux[peak_main_idx], r_values_fit, g_values_fit, parameter_values)

    #         new_red_chi_squared = reduced_chi_squared(parameter_values) 
    #         plot_best_fit_light_curve(SN_id, new_red_chi_squared, "r", "g", time, flux, fluxerr,  
    #                                   r_values, g_values, time_fit, flux_fit, save_fig = False)

    #         # print("Reduced chi-square:", new_red_chi_squared)
    #         if np.abs(red_chi_squared - new_red_chi_squared) > 0.01:
    #             red_chi_squared = new_red_chi_squared
    #             mean = parameter_values
    #             std = parameter_sigma
    #         else:
    #             done = False

    # # best_parameters_idx = np.argmin(np.abs(red_chi_squared - 1))
    # # best_parameters = estimated_parameters[best_parameters_idx]
    # np.save(f"ZTF_analytical_params/{SN_id}/best_parameters.npy", parameter_values)
        
    # time_fit = np.concatenate((np.linspace(time.min(), time.max(), 1000), np.linspace(time.min(), time.max(), 1000))) + time[peak_main_idx]
    # r_values_fit = np.arange(1000)
    # g_values_fit = np.arange(1000) + 1000
    # flux_fit = light_curve_approximation(time_fit - time[peak_main_idx], flux[peak_main_idx], r_values_fit, g_values_fit, parameter_values)

    # plot_best_fit_light_curve(SN_id, red_chi_squared, "r", "g", time, flux, fluxerr, 
    #                           r_values, g_values, time_fit, flux_fit, save_fig = False)
    

# %%

### Retrieve parameters 

ztf_parameters = []
no_plot = []

for SN_id in ztf_id:

    if os.path.isdir(f"ZTF_analytical_params/{SN_id}"):

        file = open(f"ZTF_analytical_params/{SN_id}/stats.json")
        
        stats = json.load(file)
        parameters = stats["modes"][0]["mean"]
        ztf_parameters.append(parameters)
        
        file.close()
    
    else:
        no_plot.append(SN_id)

ztf_parameters = np.array(ztf_parameters)
# %%

no_plot_idx = np.where(np.isin(ztf_id, no_plot) == True)
plot_ztf_types = np.delete(ztf_types, no_plot_idx)
plot_ztf_types_name = np.delete(ztf_types_name, no_plot_idx)

Ia_CSM_idx = (plot_ztf_types_name == "Ia_CSM")

parameters = ["A_r", "t_0_r", "t_rise_r", "gamma_r", "beta_r", "t_fall_r", "error_r", \
              "A_g", "t_0_g", "t_rise_g", "gamma_g", "beta_g", "t_fall_g", "error_g"]

important_parameters = ["t_rise_r", "gamma_r", "beta_r", "t_fall_r", \
                        "t_rise_g", "gamma_g", "beta_g", "t_fall_g"]

# %%

for idx_1 in range(6):
    for idx_2 in range(idx_1 + 1, 7): 

        plt.scatter(ztf_parameters[~Ia_CSM_idx, idx_1], ztf_parameters[~Ia_CSM_idx, idx_2], c = "tab:orange", label = "SNe IIn")
        plt.scatter(ztf_parameters[Ia_CSM_idx, idx_1], ztf_parameters[Ia_CSM_idx, idx_2], c = "tab:blue", label = "SNe Ia-CSM")

        plt.xlabel(parameters[idx_1], fontsize = 13)
        plt.ylabel(parameters[idx_2], fontsize = 13)
        plt.title(f"Parameter distribution in the ZTF r-filter.")
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()

# %%

for idx_1 in range(7):
    plt.hist(ztf_parameters[~Ia_CSM_idx, idx_1], bins = 20, color = "tab:orange", label = "SNe IIn")
    plt.hist(ztf_parameters[Ia_CSM_idx, idx_1], bins = 20, color = "tab:blue", label = "SNe Ia-CSM")

    plt.xlabel(parameters[idx_1], fontsize = 13)
    plt.ylabel("N", fontsize = 13)
    plt.title(f"Parameter distribution in the ZTF r-filter.")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()
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
