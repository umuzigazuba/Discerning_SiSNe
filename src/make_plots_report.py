# %%

from data_processing import ztf_load_data, atlas_load_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks, augment_data, find_extrema, isolate_second_peak, \
                                 parameter_bounds_gaussian, second_peak, second_peak_fit, find_parameters_one_peak

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["text.usetex"] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

### Plot the influence of each parameter to the one-peak model of Superphot+

time, flux, fluxerr, filters = atlas_load_data("ATLAS18qjm")
f1, f2 = "o", "c"
parameter_values = np.load(f"../data/best-fit/ATLAS/one_peak/ATLAS18qjm_parameters_OP.npy")

filter_f1 = np.where(filters == f1)
filter_f2 = np.where(filters == f2)

peak_main_idx = np.argmax(flux[filter_f1])
peak_time = np.copy(time[peak_main_idx])
peak_flux = np.copy(flux[peak_main_idx])

time -= peak_time

amount_fit = 100

time_fit = np.concatenate((np.linspace(time.min() - 85, time.max() + 85, amount_fit), np.linspace(time.min() - 85, time.max() + 85, amount_fit)))
f1_values_fit = np.arange(amount_fit)
f2_values_fit = np.arange(amount_fit) + amount_fit

flux_fit = light_curve_one_peak(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

rise_parameter = peak_flux * (10 ** parameter_values[0]) / (1 + np.exp(-(time_fit  + peak_time - (parameter_values[1] + peak_time)) / (10 ** parameter_values[2])))
fall_parameter = peak_flux * (10 ** parameter_values[0]) * (np.exp( ((10 ** parameter_values[0]) - (time_fit  + peak_time - (parameter_values[1] + peak_time))) / (10 ** parameter_values[5]))) 
beta = peak_flux * (10 ** parameter_values[0]) * (1 - parameter_values[4] * (time_fit  + peak_time - (parameter_values[1] + peak_time)))

r_values = np.where(filters == f1)
plt.errorbar(time[r_values] + peak_time, flux[r_values], yerr = fluxerr[r_values], fmt = "o", markersize = 4, capsize = 2, color = "dimgrey", label = f"Data {f1}-band")
plt.plot(time_fit[f1_values_fit] + peak_time, flux_fit[f1_values_fit], linestyle = "--", linewidth = 1.5, color = "dimgrey", label = f"Best fit {f1}-band")

plt.axvline(parameter_values[1] + peak_time, linewidth = 1.7, linestyle = "-.", color = colours["blue"])
plt.text(55670, 315, r"$\mathbf{t_{0}}$", fontsize = "large", color = colours["blue"])

plt.axvline(10 ** (parameter_values[3]) + parameter_values[1] + peak_time, linewidth = 1.7, linestyle = "-.", color = colours["orange"])
plt.text(55702, 515, r"$\mathbf{t_{0}} + \mathbf{\gamma}$", fontsize = "large", color = colours["orange"])

plt.plot(time_fit[f1_values_fit] + peak_time, beta[f1_values_fit], linewidth = 1.7, linestyle = (5, (10, 3)), color = colours["red"])
plt.text(55712, 315, r"$ \mathbf{A (1 - \beta(1 - t_0))}$", fontsize = "large", color = colours["red"])

plt.plot(time_fit[f1_values_fit] + peak_time, rise_parameter[f1_values_fit], linewidth = 2, linestyle = "-", color = colours["purple"])
plt.text(55610, 90, r"$ \mathbf{\frac{\sqrt{A}}{1 + \mathbf{exp}[ - \frac{t - t_{0}}{\mathbf{\tau _{rise}}}] }}$", fontsize = "x-large", color = colours["purple"])          

plt.plot(time_fit[f1_values_fit] + peak_time, fall_parameter[f1_values_fit], linewidth = 2, linestyle = "-", color = colours["green"])
plt.text(55750, 90, r"$ \mathbf{\sqrt{A}\times \mathbf{exp}[ \frac{\mathbf{\gamma} - (t - t_{0})}{\mathbf{\tau _{fall}}}]}$", fontsize = "large", color = colours["green"])

plt.xlabel("Modified Julian Date", fontsize = 13)
plt.ylabel("Flux $(\mathrm{\mu Jy})$", fontsize = 13)
plt.title(f"Light curve of ATLAS18qjm.")
plt.grid(alpha = 0.3) 
plt.xlim([55580, 55830])
plt.ylim([-25, 650])
plt.legend()
plt.savefig(f"../plots/best-fit/explanation_fitting_parameters.png", dpi = 300, bbox_inches = "tight")
plt.show()

# %%

### Plot the different steps of the two-peak model fitting process 

# time, flux, fluxerr, filters = ztf_load_data("ZTF23aansdlc")
# f1, f2 = "r", "g"
# parameter_values = np.load(f"../data/best-fit/ZTF/two_peaks/ZTF23aansdlc_parameters_TP.npy")

# f1_values = np.where(filters == f1)
# f2_values = np.where(filters == f2)

# peak_main_idx = np.argmax(flux[f1_values])
# peak_time = np.copy(time[peak_main_idx])
# peak_flux = np.copy(flux[peak_main_idx])

# time -= peak_time

# amount_fit = 100

# time_fit = np.concatenate((np.linspace(time.min() - 30, time.max() + 30, amount_fit), np.linspace(time.min() - 30, time.max() + 30, amount_fit)))
# f1_values_fit = np.arange(amount_fit)
# f2_values_fit = np.arange(amount_fit) + amount_fit

# flux_fit = light_curve_two_peaks(time_fit, parameter_values, peak_flux, f1_values_fit, f2_values_fit)

# amount_aug = 65

# time_aug, flux_aug, fluxerr_aug, filters_aug = augment_data("ZTF", amount_aug)

# f1_values_aug = np.where(filters_aug == f1)
# f2_values_aug = np.where(filters_aug == f2)

# extrema_f1, peak_width_f1, extrema_f2, peak_width_f2 = find_extrema(time_aug, flux_aug, fluxerr_aug, f1_values_aug, f2_values_aug)
# time_peak, flux_peak, fluxerr_peak, filters_peak = isolate_second_peak(f1, extrema_f1, peak_width_f1, extrema_f2, peak_width_f2, 
#                                                                        time_aug, f1_values_aug, f2_values_aug)

# f1_values_peak = np.where(filters_peak == f1)
# f2_values_peak = np.where(filters_peak == f2)

# # Inital guess of the parameters (amplitude, mean, standard deviation)
# # If in both filters a second peak is detected
# if f1 in filters and extrema_f1[2] != -1 and f2 in filters and extrema_f2[2] != -1:
#     guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 
#                                 flux_aug[f2_values_aug][extrema_f2][2] / flux_aug[f1_values_aug][extrema_f1][2], 
#                                 time_aug[f2_values_aug][extrema_f2][2] / time_aug[f1_values_aug][extrema_f1][2], 
#                                 peak_width_f2 / peak_width_f1])

# # # If only in one peak a second peak is detected
# # else:
# #     if f1 in filters and extrema_f1[2] != -1:
# #         guess_parameters = np.array([1, time_aug[f1_values_aug][extrema_f1][2], peak_width_f1, 1, 1, 1])

# #     elif f2 in filters and extrema_f2[2] != -1:
# #         guess_parameters = np.array([1, time_aug[f2_values_aug][extrema_f2][2], peak_width_f2, 1, 1, 1])

# down_bound = parameter_bounds_gaussian[0]
# up_bound = parameter_bounds_gaussian[1]

# second_peak_parameters, _ = curve_fit(second_peak, time_peak, flux_peak, p0 = guess_parameters, sigma = fluxerr_peak, bounds = parameter_bounds_gaussian)
# flux_peak = second_peak_fit(time, *second_peak_parameters)

# flux -= flux_peak

# parameter_values_OP, _ = find_parameters_one_peak("ZTF23aansdlc", "ZTF")
# flux_OP_fit = light_curve_one_peak(time_fit, parameter_values_OP, peak_flux, f1_values_fit, f2_values_fit)

# flux += flux_peak

# time += peak_time
# time_fit += peak_time
# time_aug += peak_time
# time_peak += peak_time

# r_values = np.where(filters == f1)
# plt.errorbar(time[r_values], flux[r_values], yerr = fluxerr[r_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = f"Residual data r-band")
# # plt.errorbar(time_peak[f1_values_peak], flux_peak[9:12], yerr = fluxerr_peak[f1_values_peak], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = f"Data second peak r-band")
# plt.plot(time_fit[f1_values_fit], flux_OP_fit[f1_values_fit], linestyle = "--", linewidth = 2, color = "tab:blue", label = f"Best fit residual data {f1}-band")
# # plt.plot(time[f1_values], flux_peak[f1_values], linewidth = 2, linestyle = ":", color = "black", label = "Best fit second peak r-band")

# # plt.fill_between(time_aug[f1_values_aug], flux_aug[f1_values_aug] - fluxerr_aug[f1_values_aug], flux_aug[f1_values_aug] + fluxerr_aug[f1_values_aug], color = "tab:blue", alpha = 0.15, label = "Error approximation r-band")
# # plt.plot(time_aug[f1_values_aug], flux_aug[f1_values_aug], linewidth = 2, color = "tab:blue", zorder = 5, label = "Flux approximation r-band")

# # plt.scatter(time_aug[f1_values_aug][extrema_f1][::2], flux_aug[f1_values_aug][extrema_f1][::2], s = 200, marker = "*", c = "tab:orange", edgecolors = "black", label = "Extrema r-band" , zorder = 10)
# # plt.scatter(time_aug[f1_values_aug][extrema_f1][1], flux_aug[f1_values_aug][extrema_f1][1], s = 200, marker = "*", c = "tab:orange", edgecolors = "black", zorder = 10)

# plt.xlabel("Modified Julian Date", fontsize = 13)
# plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
# plt.title(f"Best fit residual light curve of ZTF23aansdlc.")
# plt.grid(alpha = 0.3) 
# # plt.xlim([-4, 54])
# # plt.ylim([-25, 425])
# plt.legend()
# # plt.savefig(f"Presentation/fitted_lightcurve_TP_residual.png", dpi = 300, bbox_inches = "tight")
# plt.show()
# %%
