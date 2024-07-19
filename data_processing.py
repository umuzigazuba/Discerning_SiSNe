# %% 

from antares_client.search import get_by_ztf_object_id
import fulu
from astropy.coordinates import SkyCoord
from dustmaps.sfd import SFDQuery
import extinction
from sklearn.gaussian_process.kernels import (RBF, Matern, 
      WhiteKernel, ConstantKernel as C)

import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["text.usetex"] = True

# %%

### ZTF ###

# Generate data
def retrieve_ztf_data(ztf_id):

    data = get_by_ztf_object_id(ztf_id)

    ra = data.ra
    dec = data.dec

    valid_data_r = np.where((data.lightcurve["ant_passband"].to_numpy() == "R") & (~np.isnan(data.lightcurve["ant_mag"].to_numpy())))[0]

    if len(valid_data_r != 0):
        time_r = data.lightcurve["ant_mjd"].to_numpy()[valid_data_r]
        mag_r = data.lightcurve["ant_mag"].to_numpy()[valid_data_r]
        magerr_r = data.lightcurve["ant_magerr"].to_numpy()[valid_data_r]
        
    else:
        time_r = np.array([])
        mag_r = np.array([])
        magerr_r = np.array([])

    valid_data_g = np.where((data.lightcurve["ant_passband"].to_numpy() == "g") & (~np.isnan(data.lightcurve["ant_mag"].to_numpy())))[0]

    if len(valid_data_g != 0):
        time_g = data.lightcurve["ant_mjd"].to_numpy()[valid_data_g]
        mag_g = data.lightcurve["ant_mag"].to_numpy()[valid_data_g]
        magerr_g = data.lightcurve["ant_magerr"].to_numpy()[valid_data_g]
        
    else:
        time_g = np.array([])
        mag_g = np.array([])
        magerr_g = np.array([])
    
    return ra, dec, time_r, mag_r, magerr_r, time_g, mag_g, magerr_g

def ztf_magnitude_to_micro_flux(magnitude, magnitude_error):
    flux = np.power(10, -0.4*(magnitude - 26.3))
    flux_error = 0.4*np.log(10)*magnitude_error*flux
    return flux, flux_error

def load_ztf_data(ztf_id):

    time = []
    flux = []
    fluxerr = []
    filters = []

    if os.path.isdir(f"Data/ZTF_data/{ztf_id}/r/"):
        ztf_data_r = np.load(f"Data/ZTF_data/{ztf_id}/r/.npy")

        time.extend(ztf_data_r[0])
        flux.extend(ztf_data_r[1])
        fluxerr.extend(ztf_data_r[2])
        filters.extend(["r"] * len(ztf_data_r[0]))

    if os.path.isdir(f"Data/ZTF_data/{ztf_id}/g/"):
        ztf_data_g = np.load(f"Data/ZTF_data/{ztf_id}/g/.npy")

        time.extend(ztf_data_g[0])
        flux.extend(ztf_data_g[1])
        fluxerr.extend(ztf_data_g[2])
        filters.extend(["g"] * len(ztf_data_g[0]))

    return np.array(time), np.array(flux), np.array(fluxerr), np.array(filters)

# Plot data
def plot_ztf_data(ztf_id, time, flux, fluxerr, filters, save_fig = False):

    if "r" in filters:
        r_values = np.where(filters == "r")
        plt.errorbar(time[r_values], flux[r_values], yerr = fluxerr[r_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = "Band: r")

    if "g" in filters:
        g_values = np.where(filters == "g")
        plt.errorbar(time[g_values], flux[g_values], yerr = fluxerr[g_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = "Band: g")

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {ztf_id}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/ZTF_lightcurves_plots/ZTF_data_{ztf_id}", dpi = 300)
        plt.close()
    else:
        plt.show()

# %%

### ATLAS ###

def retrieve_atlas_data(object_type, atlas_id, discovery_dates, passband):

    data = np.loadtxt(f"Data/ATLAS_data/{object_type}/{atlas_id}/{atlas_id}.{passband}.1.00days.lc.txt", skiprows = 1, usecols = (0, 2, 3))
    valid_data_idx = np.where((~np.isnan(data[:, 0])) & (data[:, 2] < 50))
    valid_data = data[valid_data_idx]

    start_date_idx = np.where(discovery_dates[:, 0] == atlas_id)
    start_date = float(discovery_dates[start_date_idx, 1][0, 0])
    last_date = valid_data[-1, 0]


    if start_date + 300 > last_date:
        end_date = last_date
    else:
        end_date = start_date + 300
    SN_dates = np.where((valid_data[:, 0] >= start_date) & (valid_data[:, 0] <= end_date))

    if len(SN_dates[0] != 0):
        time = valid_data[SN_dates, 0]
        flux = valid_data[SN_dates, 1]
        fluxerr = valid_data[SN_dates, 2]
            
        return time[0], flux[0], fluxerr[0]
    
    return [], [], []


def plot_atlas_data(atlas_id, time_c, flux_c, fluxerr_c, time_o, flux_o, fluxerr_o, save_fig = False):

    if len(time_c) != 0:
        plt.errorbar(time_c, flux_c, yerr = fluxerr_c, fmt = "o", markersize = 4, capsize = 2, color = "blue", label = "Band: c")

    if len(time_o) != 0:
        plt.errorbar(time_o, flux_o, yerr = fluxerr_o, fmt = "o", markersize = 4, capsize = 2, color = "orange", label = "Band: o")

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {atlas_id}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/ATLAS_lightcurves_plots/ATLAS_data_{atlas_id}", dpi = 300)
        plt.close()
    else:
        plt.show()

# %%

### Light curve approximation ###

def data_augmentation(survey, time, flux, fluxerr, filters, augmentation_type):

    if survey == "ZTF":
        passband2lam = {'r': np.log10(6366.38), 'g': np.log10(4746.48)}

    elif survey == "ATLAS":
        passband2lam = {'o': np.log10(6629.82), 'c': np.log10(5182.42)}
    
    else:
        print("ERROR: the options for survey are \"ZTF\" and \"ATLAS\".")
        return None
    
    passbands = filters
    if augmentation_type == "GP":
        augmentation = fulu.GaussianProcessesAugmentation(passband2lam, C(1.0)*Matern() * RBF([1, 1]) + Matern() + WhiteKernel())

    elif augmentation_type == "MLP":
        augmentation = fulu.MLPRegressionAugmentation(passband2lam)
    
    elif augmentation_type == "NF":
        augmentation = fulu.NormalizingFlowAugmentation(passband2lam)
    
    elif augmentation_type == "BNN":
        augmentation = fulu.BayesianNetAugmentation(passband2lam)
    
    else:
        print("ERROR: the options for augmentation_type are \"GP\", \"MLP\", \"NF\"and \"BNN\".")
        return None

    augmentation.fit(time, flux, fluxerr, passbands)

    return passbands, passband2lam, augmentation

def plot_data_augmentation(SN_id, passbands, passband2lam, augmentation_type, time, flux,
                           fluxerr, time_aug, flux_aug, flux_err_aug, passband_aug, ax): 

    plot = fulu.LcPlotter(passband2lam)
    plot.plot_one_graph_all(t = time, flux = flux, flux_err = fluxerr, passbands = passbands,
                            t_approx = time_aug, flux_approx = flux_aug,
                            flux_err_approx = flux_err_aug, passband_approx = passband_aug, ax = ax,
                            title = f"Augmented light curve of SN {SN_id} using {augmentation_type}.")

# %%

# ZTF data

ztf_id_sn_Ia_CSM= np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
ztf_id_sn_IIn= np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")
        
ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))
ztf_types = np.concatenate((np.zeros(len(ztf_id_sn_Ia_CSM)), np.ones(len(ztf_id_sn_IIn))))

# ATLAS data

atlas_id_sn_Ia_CSM = np.loadtxt("Data/ATLAS_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
atlas_id_sn_IIn = np.loadtxt("Data/ATLAS_ID_SNe_IIn", delimiter = ",", dtype = "str")

atlas_id = np.concatenate((atlas_id_sn_Ia_CSM, atlas_id_sn_IIn))
atlas_types = np.concatenate((np.zeros(len(atlas_id_sn_Ia_CSM)), np.ones(len(atlas_id_sn_IIn))))
discovery_dates = data = np.loadtxt("Data/ATLAS_data/sninfo.txt", skiprows = 1, usecols = (0, 3), dtype = "str")

# %%

### Data processing ###

## Small light curves + light curve clipping: 

def OLD_plot_ztf_data(ztf_id, time_g, flux_g, fluxerr_g, time_r, flux_r, fluxerr_r, save_fig = False):

    if len(time_r) != 0:
        plt.errorbar(time_r, flux_r, yerr = fluxerr_r, fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = "Band: r")

    if len(time_g) != 0:
        plt.errorbar(time_g, flux_g, yerr = fluxerr_g, fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = "Band: g")

    plt.xlabel("Modified Julian Date", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {ztf_id}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/ZTF_lightcurves_plots/ZTF_data_{ztf_id}", dpi = 300)
        plt.close()
    else:
        plt.show()


def get_magnitud_extinction(magnitude, ra, dec, wavelength):

    
    sfd = SFDQuery()
    coordinates = SkyCoord(ra, dec, frame = "icrs", unit = "deg")

    MW_EBV = sfd(coordinates)
    Av = 2.742 * MW_EBV

    wavelength = 1.0 / (0.0001 * np.array([wavelength]))

    delta_mag = extinction.fm07(wavelength, Av, unit = "invum")

    return magnitude - delta_mag

def data_processing():

    r_wavelength = 6173.23
    g_wavelength = 4741.64

    for SN_id in ztf_id:

        ra, dec, time_r, mag_r, magerr_r, time_g, mag_g, magerr_g = retrieve_ztf_data(SN_id)

        mag_r = get_magnitud_extinction(mag_r, ra, dec, r_wavelength)
        mag_g = get_magnitud_extinction(mag_g, ra, dec, g_wavelength)

        flux_g, fluxerr_g = ztf_magnitude_to_micro_flux(mag_g, magerr_g)
        flux_r, fluxerr_r = ztf_magnitude_to_micro_flux(mag_r, magerr_r)

        # Light curve clipping 
        
        if len(time_g) != 0:
            peak_idx_g =  0

            if len(time_g) > 5:
                while (flux_g[peak_idx_g] < flux_g[peak_idx_g + 1 : peak_idx_g + 3]).all():
                    peak_idx_g += 1
            else:
                peak_idx_g = np.argmax(flux_g)

            end_idx_g = len(time_g) - peak_idx_g
            pts_to_delete_g = 0

            if peak_idx_g != len(time_g) - 1:

                peak_slope_g = (flux_g[-1] - flux_g[peak_idx_g])/(time_g[-1] - time_g[peak_idx_g])

                for idx in range(2, end_idx_g):

                    last_idx_g = -1 * idx
                    slope_g = (flux_g[last_idx_g] - flux_g[-1])/(time_g[last_idx_g] - time_g[-1])

                    if np.abs(slope_g) < 0.2 * np.abs(peak_slope_g):
                        pts_to_delete_g = idx

                if pts_to_delete_g > 0:

                    time_g = time_g[: -pts_to_delete_g]
                    flux_g = flux_g[: -pts_to_delete_g]
                    fluxerr_g = fluxerr_g[: -pts_to_delete_g]
        
        if len(time_r) != 0:
            peak_idx_r =  0

            if len(time_r) > 5:
                while (flux_r[peak_idx_r] < flux_r[peak_idx_r + 1 : peak_idx_r + 3]).all():
                    peak_idx_r += 1
            else:
                peak_idx_r = np.argmax(flux_r)

            end_idx_r = len(time_r) - peak_idx_r
            pts_to_delete_r = 0

            if peak_idx_r != len(time_r) - 1:

                peak_slope_r = (flux_r[-1] - flux_r[peak_idx_r])/(time_r[-1] - time_r[peak_idx_r])

                for idx in range(2, end_idx_r):

                    last_idx_r = -1 * idx
                    slope_r = (flux_r[last_idx_r] - flux_r[-1])/(time_r[last_idx_r] - time_r[-1])

                    if np.abs(slope_r) < 0.2 * np.abs(peak_slope_r):
                        pts_to_delete_r = idx

                if pts_to_delete_r > 0:

                    time_r = time_r[: -pts_to_delete_r]
                    flux_r = flux_r[: -pts_to_delete_r]
                    fluxerr_r = fluxerr_r[: -pts_to_delete_r]

        # Signal-to-noise and Brightness variability
        if len(time_g) != 0:
            SNR_g = flux_g / fluxerr_g
            good_SNR_g = np.where(SNR_g > 3)[0]

            max_flux_g = np.max(flux_g) - np.min(flux_g)
            mean_fluxerr_g = np.mean(fluxerr_g)

            std_flux_g = np.std(flux_g)

            if (((len(good_SNR_g) or len(time_g)) < 5) or (max_flux_g < 3. * mean_fluxerr_g)) or (std_flux_g < mean_fluxerr_g):
                time_g = []
                flux_g = []
                fluxerr_g = []

        if len(time_r) != 0:
            SNR_r = flux_r / fluxerr_r
            good_SNR_r = np.where(SNR_r > 3)[0]

            max_flux_r = np.max(flux_r) - np.min(flux_r)
            mean_fluxerr_r = np.mean(fluxerr_r)

            std_flux_r = np.std(flux_r)

            if (((len(good_SNR_r) or len(time_r)) < 5) or (max_flux_r < 3. * np.mean(fluxerr_r))) or (std_flux_r < np.mean(fluxerr_r)):
                time_r = []
                flux_r = []
                fluxerr_r = []

        if len(time_g) > 0 or len(time_r) > 0:
            OLD_plot_ztf_data(SN_id, time_g, flux_g, fluxerr_g, time_r, flux_r, fluxerr_r, save_fig = True)

        if len(time_g) > 0:

            SN_data_g = np.stack((time_g, flux_g, fluxerr_g))

            os.makedirs(f"Data/ZTF_data/{SN_id}/g/", exist_ok = True)
            np.save(f"Data/ZTF_data/{SN_id}/g/", SN_data_g)

        if len(time_r) > 0:
            SN_data_r = np.stack((time_r, flux_r, fluxerr_r))

            os.makedirs(f"Data/ZTF_data/{SN_id}/r/", exist_ok = True)
            np.save(f"Data/ZTF_data/{SN_id}/r/", SN_data_r)

# %%

def test():

    # ZTF data

    ztf_id_sn_Ia_CSM= np.loadtxt("Data/ZTF_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
    ztf_id_sn_IIn= np.loadtxt("Data/ZTF_ID_SNe_IIn", delimiter = ",", dtype = "str")
            
    ztf_id = np.concatenate((ztf_id_sn_Ia_CSM, ztf_id_sn_IIn))
    ztf_types = np.concatenate((np.zeros(len(ztf_id_sn_Ia_CSM)), np.ones(len(ztf_id_sn_IIn))))

    # ATLAS data

    atlas_id_sn_Ia_CSM = np.loadtxt("Data/ATLAS_ID_SNe_Ia_CSM", delimiter = ",", dtype = "str")
    atlas_id_sn_IIn = np.loadtxt("Data/ATLAS_ID_SNe_IIn", delimiter = ",", dtype = "str")

    atlas_id = np.concatenate((atlas_id_sn_Ia_CSM, atlas_id_sn_IIn))
    atlas_types = np.concatenate((np.zeros(len(atlas_id_sn_Ia_CSM)), np.ones(len(atlas_id_sn_IIn))))
    discovery_dates = data = np.loadtxt("Data/ATLAS_data/sninfo.txt", skiprows = 1, usecols = (0, 3), dtype = "str")

    # ZTF 
    for SN_id in ztf_id:

        time_g, mag_g, magerr_g = retrieve_ztf_data(SN_id, "g")
        time_r, mag_r, magerr_r = retrieve_ztf_data(SN_id, "R")

        flux_g, fluxerr_g = ztf_magnitude_to_micro_flux(mag_g, magerr_g)
        flux_r, fluxerr_r = ztf_magnitude_to_micro_flux(mag_r, magerr_r)

        plot_ztf_data(SN_id, time_g, flux_g, fluxerr_g, time_r, flux_r, fluxerr_r, save_fig = True)

    # ATLAS
    # time_c, flux_c, fluxerr_c = retrieve_atlas_data("SN_IIn", atlas_id_sn_Ia_CSM[3], discovery_dates, "c")
    # time_o, flux_o, fluxerr_o = retrieve_atlas_data("SN_IIn", atlas_id_sn_Ia_CSM[3], discovery_dates, "o")

    # plot_atlas_data(atlas_id_sn_Ia_CSM[3], time_c, flux_c, fluxerr_c, time_o, flux_o, fluxerr_o)

    # Data augmentation
    # time, flux, fluxerr, passbands, passband2lam, augmentation = data_augmentation("ATLAS", time_c, flux_c, fluxerr_c,
    #                                                                             time_o, flux_o, fluxerr_o, "MLP")

    # approx_peak_idx = np.argmax(flux)
    # approx_peak_time = time[approx_peak_idx]

    # time_aug, flux_aug, flux_err_aug, passband_aug = augmentation.augmentation(approx_peak_time - 100, approx_peak_time + 250, n_obs = 1000)

    # plot_data_augmentation(atlas_id_sn_Ia_CSM[1], passbands, passband2lam, "MLP",
    #                     time, flux, fluxerr, time_aug, flux_aug, flux_err_aug, passband_aug)
    
    # time, flux, fluxerr, passbands, passband2lam, augmentation = data_augmentation("ATLAS", time_c, flux_c, fluxerr_c,
    #                                                                             time_o, flux_o, fluxerr_o, "GP")

    # approx_peak_idx = np.argmax(flux)
    # approx_peak_time = time[approx_peak_idx]

    # time_aug, flux_aug, flux_err_aug, passband_aug = augmentation.augmentation(time.min(), time.max(), n_obs = 1000)

    # plot_data_augmentation("2020ywx", passbands, passband2lam, "GP",
    #                     time, flux, fluxerr, time_aug, flux_aug, flux_err_aug, passband_aug)
    
    time, flux, fluxerr, passbands, passband2lam, augmentation = data_augmentation("ZTF", time_g, flux_g, fluxerr_g,
                                                                                time_r, flux_r, fluxerr_r, "GP")

    approx_peak_idx = np.argmax(flux)
    approx_peak_time = time[approx_peak_idx]

    time_aug, flux_aug, flux_err_aug, passband_aug = augmentation.augmentation(approx_peak_time - 100, approx_peak_time + 250, n_obs = 1000)

    plot_data_augmentation(ztf_id[1], passbands, passband2lam, "GP",
                        time, flux, fluxerr, time_aug, flux_aug, flux_err_aug, passband_aug)
    
    # time, flux, fluxerr, passbands, passband2lam, augmentation = data_augmentation("ZTF", time_g, flux_g, fluxerr_g,
    #                                                                             time_r, flux_r, fluxerr_r, "NF")

    # approx_peak_idx = np.argmax(flux)
    # approx_peak_time = time[approx_peak_idx]

    # time_aug, flux_aug, flux_err_aug, passband_aug = augmentation.augmentation(approx_peak_time - 100, approx_peak_time + 250, n_obs = 1000)

    # plot_data_augmentation(ztf_id[1], passbands, passband2lam, "NF",
    #                     time, flux, fluxerr, time_aug, flux_aug, flux_err_aug, passband_aug)

# %%
    
if __name__ == '__main__':
    test()

# %%
