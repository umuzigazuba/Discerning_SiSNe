# %% 

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz

from dustmaps.config import config
config['data_dir']= "../utils/"
from dustmaps.sfd import SFDQuery
from extinction import fm07, remove

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

### ZTF ###

def ztf_retrieve_data(ztf_name):

    """ 
    Retrieve the raw data for a given ZTF SN

    Parameters:
        ztf_name (str): The internal ZTF name of the SN 

    Outputs: 
        ztf_data (numpy.ndarray): Raw data generated from the ZTF forced photometry service
    """

    with open(f"../data/raw/ZTF/{ztf_name}.txt") as f:

        # Ignore lines with parameter explenation
        lines = (line for line in f if not line.startswith('#'))

        # Ignore line with parameter names
        ztf_data = np.genfromtxt(lines, skip_header = 1, missing_values = "null", dtype = "str")

    return ztf_data

def calculate_airmass(ra, dec, time):

    """ 
    Calculates the airmass from the Palomar telescope in the direction of the SN 
    Used to determine the necessary quality cuts

    Parameters:
        ra (float): Right ascension of the SN 
        dec (float): Declination of the SN 
        time (numpy.ndarray): Modified Julian Dates of the observations. 

    Outputs: 
        airmass (numpy.ndarray): Airmass in the direction of the SN. 
    """

    # Telescope location
    palomar = EarthLocation.of_site('Palomar')

    # SN
    target = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))
    time = Time(time, format = "jd")

    target_in_altaz = target.transform_to(AltAz(obstime = time, location = palomar))
    airmass = target_in_altaz.secz.value

    return airmass

def ztf_remove_noisy_data(ra, dec, ccd_threshold, ztf_data):

    """ 
    Remove noisy and "bad" observations from the ZTF light curves

    Parameters:
        ztf_name (str): The internal ZTF name of the SN 

    Outputs: 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: ZTF_g, ZTF_r and ZTF_i
        zero_point (numpy.ndarray): Zeropoint for the science image
        reduced_chi_squared (numpy.ndarray): Reduced chi-squared value of the PSI-fit
    """

    # Supernova data
    time = np.array(ztf_data[:, 22]) # in Julian Date
    flux = np.array(ztf_data[:, 24])
    fluxerr = np.array(ztf_data[:, 25])
    filters = np.array(ztf_data[:, 4])

    # Parameters used to identify noisy/bad data
    infobitssci = np.array(ztf_data[:, 6]).astype(np.float32)
    scisigpix = np.array(ztf_data[:, 9]).astype(np.float32)
    sciinpseeing = np.array(ztf_data[:, 7]).astype(np.float32)
    status = np.array(ztf_data[:, -1]).astype(str)

    # CCD Camera and number of calibrators
    ccd = np.array(ztf_data[:, 2]).astype(np.float32)
    ccd_amplifier = np.array(ztf_data[:, 3]).astype(np.float32)
    ncalmatches = np.array(ztf_data[:, 15]).astype(np.float32)

    zero_point = np.array(ztf_data[:, 10]).astype(np.float32)
    zero_point_rms = np.array(ztf_data[:, 12]).astype(np.float32)
    reduced_chi_squared = np.array(ztf_data[:, 27]).astype(np.float32)

    # Filter out bad processing epochs
    bad_observations = np.where(((status != "0") & (status != "56") & (status != "57") & (status != "62") & (status != "65")))[0]

    # Filter out data with missing flux measurements
    missing_data = np.where(flux == "null")[0]

    # Filter out data of a bad quality (possibly contaminated by clouds or the moon)
    bad_infobitssci = np.where(infobitssci > 0)[0]
    bad_scisigpix = np.where(scisigpix > 25)[0]
    bad_sciinpseeing = np.where(sciinpseeing > 4)[0]

    # Filter out badly calibrated epochs 
    airmass = calculate_airmass(ra, dec, time)
    rcid = 4 * (ccd - 1) + ccd_amplifier - 1

    quality_cuts_f1 = np.where((filters == "ZTF_r") & ((zero_point > (26.65 - 0.15 * airmass)) | (zero_point_rms > 0.05) | (ncalmatches < 120) | (zero_point < (ccd_threshold["r"].iloc[rcid].to_numpy() - 0.15 * airmass))))[0]
    quality_cuts_f2 = np.where((filters == "ZTF_g") & ((zero_point > (26.7 - 0.2 * airmass)) | (zero_point_rms > 0.06) | (ncalmatches < 80) | (zero_point < (ccd_threshold["g"].iloc[rcid].to_numpy() - 0.2 * airmass))))[0]

    # Delete noisy/bad data 
    indices_to_delete = np.sort(np.unique(np.concatenate((bad_observations, missing_data, bad_infobitssci, bad_scisigpix, bad_sciinpseeing, quality_cuts_f1, quality_cuts_f2))))

    time = np.delete(time, indices_to_delete).astype(np.float32) - 2400000.5       # in Modified Julian Date
    flux = np.delete(flux, indices_to_delete).astype(np.float32) / 10       # in microJansky
    fluxerr = np.delete(fluxerr, indices_to_delete).astype(np.float32) / 10       # in microJansky
    filters = np.delete(filters, indices_to_delete)

    zero_point = np.delete(zero_point, indices_to_delete)
    reduced_chi_squared = np.delete(reduced_chi_squared, indices_to_delete)

    return time, flux, fluxerr, filters, zero_point, reduced_chi_squared

def check_flux_uncertainties(fluxerr, reduced_chi_squared):

    """ 
    Check and, if necessary, adjust the flux errors for systematic errors

    Parameters:
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        reduced_chi_squared (numpy.ndarray): Reduced chi-squared value of the PSI-fit

    Outputs: 
        fluxerr (numpy.ndarray): Rescaled one-sigma error on the differential flux of observations in microJansky
    """

    average_reduced_chi_squared = np.mean(reduced_chi_squared)

    if not np.isclose(average_reduced_chi_squared, 1, 0.5):

        fluxerr *= np.sqrt(average_reduced_chi_squared)

    return fluxerr

def ztf_micro_flux_to_magnitude(flux, fluxerr, zero_point):

    """ 
    Calculate the magnitude from the flux measurements in microJansky

    Parameters:
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        zero_point (numpy.ndarray): Zeropoint for the science image

    Outputs: 
        magnitude (numpy.ndarray): (Upper limit on the) magnitude of the observations
        magnitudeerr (numpy.ndarray): One-sigma error on the magnitude
    """

    magnitude = np.empty(len(flux))
    magnitudeerr = np.empty(len(flux))

    # Select confident observations with a signal-to-noise ratio above a threshold
    confident_detections = flux / fluxerr > 3

    # Confident detections
    # Multiply the flux by 10 to return to the original scale
    magnitude[confident_detections] = zero_point[confident_detections] - 2.5 * np.log10(10 * flux[confident_detections])
    magnitudeerr[confident_detections] = 1.0857 * fluxerr[confident_detections] / flux[confident_detections]

    # Calculate the upper limit for non-detections
    magnitude[~confident_detections] = zero_point[~confident_detections] - 2.5 * np.log10(10 * 5 * fluxerr[~confident_detections])
    magnitudeerr[~confident_detections] = 0

    return magnitude, magnitudeerr

def ztf_load_data(ztf_name):

    """ 
    Load the processed ZTF light curve data 

    Parameters:
        ztf_name (str): The internal ZTF name of the SN 

    Outputs: 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: g, r
    """

    ztf_data = np.load(f"../data/processed/ZTF/{ztf_name}_data.npy")

    time = ztf_data[:, 0]
    flux = ztf_data[:, 1]
    fluxerr = ztf_data[:, 2]
    filters = ztf_data[:, 3]

    return np.array(time).astype(np.float32), np.array(flux).astype(np.float32), \
           np.array(fluxerr).astype(np.float32), np.array(filters)

def ztf_plot_data(ztf_name, time, flux, fluxerr, filters, save_fig = False):

    """ 
    Plot the processed light curve data in every filter

    Parameters:
        ztf_name (str): The internal ZTF name of the SN 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: g, r
        save_fig (boolean): If True, the plot is saved

    Outputs: 
        None
    """

    r_values = np.where(filters == "r")
    g_values = np.where(filters == "g")
    
    plt.errorbar(time[r_values], flux[r_values], yerr = fluxerr[r_values], fmt = "o", markersize = 4, capsize = 2, color = colours["blue"], label = "Band: r")
    plt.errorbar(time[g_values], flux[g_values], yerr = fluxerr[g_values], fmt = "o", markersize = 4, capsize = 2, color = colours["orange"], label = "Band: g")

    plt.xlabel("Modified Julian Date (days)", fontsize = 13)
    plt.ylabel("Differential flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {ztf_name}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"../plots/processed/ZTF/Data_points_{ztf_name}", dpi = 300, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

# %%

### ATLAS ###

def atlas_retrieve_data(atlas_name):

    """ 
    Retrieve the raw data for a given ATLAS SN

    Parameters:
        atlas_name (str): The TNS name of the ATLAS SN 

    Outputs: 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: o, c
    """

    # Load stacked and cleaned data 
    atlas_data =  np.loadtxt(f"../data/raw/ATLAS/cleaned_and_stacked/{atlas_name}_atlas_fp_stacked_2_days.txt", delimiter = ",", dtype = str)

    # Skip first row with the parameter names
    time = atlas_data[1:, 0].astype(np.float32)
    flux = atlas_data[1:, 1].astype(np.float32)
    fluxerr = atlas_data[1:, 2].astype(np.float32)
    filters = atlas_data[1:, 3].astype(str)

    return time, flux, fluxerr, filters

def atlas_remove_noisy_data(time, flux, fluxerr, filters):

    """ 
    Remove noisy observations from the ATLAS light curves

    Parameters:
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: o, c

    Outputs: 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: o, c
    """

    # Filter out observations with large uncertainties
    delete_flux_error = np.where(fluxerr > 40)

    time = np.delete(time, delete_flux_error)
    flux = np.delete(flux, delete_flux_error)
    fluxerr = np.delete(fluxerr, delete_flux_error)
    filters = np.delete(filters, delete_flux_error)

    return time, flux, fluxerr, filters

def atlas_micro_flux_to_magnitude(flux, fluxerr):

    """ 
    Calculate the magnitude from the flux measurements in microJansky

    Parameters:
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky

    Outputs: 
        magnitude (numpy.ndarray): Magnitude of the observations
        magnitudeerr (numpy.ndarray): One-sigma error on the magnitude
    """

    magnitude = -2.5 * np.log10(flux) + 23.9
    magnitudeerr = np.abs(-2.5 * (fluxerr / (flux * np.log(10))))

    return magnitude, magnitudeerr

def atlas_load_data(atlas_name):

    """ 
    Load the processed ATLAS light curve data 

    Parameters:
        atlas_name (str): The internal ATLAS name of the SN 

    Outputs: 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: o, c
    """


    atlas_data = np.load(f"../data/processed/ATLAS/{atlas_name}_data.npy")

    time = atlas_data[:, 0]
    flux = atlas_data[:, 1]
    fluxerr = atlas_data[:, 2]
    filters = atlas_data[:, 3]

    return np.array(time).astype(np.float32), np.array(flux).astype(np.float32), \
           np.array(fluxerr).astype(np.float32), np.array(filters)


def atlas_plot_data(atlas_name, time, flux, fluxerr, filters, save_fig = False):

    """ 
    Plot the processed light curve data in every filter

    Parameters:
        atlas_name (str): The internal ATLAS name of the SN 
        time (numpy.ndarray): Modified Julian Date of observations 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        fluxerr (numpy.ndarray): One-sigma error on the differential flux of observations in microJansky
        filters (numpy.ndarray): Filter used for the observation: o, c
        save_fig (boolean): If True, the plot is saved

    Outputs: 
        None
    """
    
    o_values = np.where(filters == "o")
    c_values = np.where(filters == "c")

    plt.errorbar(time[o_values], flux[o_values], yerr = fluxerr[o_values], fmt = "o", markersize = 4, capsize = 2, color = colours["blue"], label = "Band: o")
    plt.errorbar(time[c_values], flux[c_values], yerr = fluxerr[c_values], fmt = "o", markersize = 4, capsize = 2, color = colours["orange"], label = "Band: c")

    plt.xlabel("Modified Julian Date (days)", fontsize = 13)
    plt.ylabel("Differential flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {atlas_name}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"../plots/processed/ATLAS/Data_points_{atlas_name}", dpi = 300, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

# %%

### General ###

# Baseline subtraction
    # Has to be after cleaning because the peak needs to be determined
    # Often highest value has large error and is probably background noise

def find_baseline(flux, f1_values, f2_values, window = 5):

    """ 
    Select baseline (= historic and future epochs) from the SN data

    Parameters:
        flux (numpy.ndarray): Differential flux of observations in microJansky
        f1_values (numpy.ndarray): Indices of observations measured in the f1 filter
        f2_values (numpy.ndarray): Indices of observations measured in the f2 filter
        window (int): Number of neighbouring points considered during one iterations

    Outputs: 
        past_and_future_epochs (dict): The indices of first and last observations belonging to the SN in both filters
    """

    past_and_future_epochs = {"Extrema f1":[], "Extrema f2":[]}

    # Past
    # Iterate over all filters
    for filter_number, filter_idx in enumerate([f1_values, f2_values]):
        flux_filter = flux[filter_idx]

        peak_idx = np.argmax(flux_filter)
        beginning_SN = 0

        first_section = flux_filter[:window]
        first_mean = np.mean(first_section)

        for idx in range(window, peak_idx, window):

            next_section = flux_filter[idx:idx + window]
            next_mean = np.mean(next_section)

            if np.abs(next_mean - first_mean) > window:
                beginning_SN = idx
                break

        past_and_future_epochs[f"Extrema f{filter_number + 1}"].append(beginning_SN)

    # Future
    for filter_number, filter_idx in enumerate([f1_values, f2_values]):

        flux_filter = flux[filter_idx]

        peak_idx = np.argmax(flux_filter)
        end_SN = -1

        last_section = flux_filter[-window:]
        last_mean = np.mean(last_section)

        for idx in range(-window, - (len(flux_filter) - peak_idx), -window):

            next_section = flux_filter[idx - window : idx]
            next_mean = np.mean(next_section)

            if np.abs(next_mean - last_mean) > window:
                end_SN = idx
                break
            
        past_and_future_epochs[f"Extrema f{filter_number + 1}"].append(end_SN)

    return past_and_future_epochs

def subtract_baseline(flux, f1_values, f2_values, past_and_future_epochs):

    """ 
    Subtract the average flux of the baseline from the differential flux

    Parameters:
        flux (numpy.ndarray): Differential flux of observations in microJansky
        f1_values (numpy.ndarray): Indices of observations measured in the f1 filter
        f2_values (numpy.ndarray): Indices of observations measured in the f2 filter
        past_and_future_epochs (dict): The indices of first and last observations belonging to the SN in both filters

    Outputs: 
        flux (numpy.ndarray): Differential flux of observations in microJansky, adjusted for the baseline
    """

    if len(past_and_future_epochs["Extrema f1"]) != 0:

        baseline_f1 = np.concatenate((flux[f1_values][:past_and_future_epochs["Extrema f1"][0]], flux[f1_values][past_and_future_epochs["Extrema f1"][1]:]))
        average_flux_baseline_f1 = np.mean(baseline_f1)

        flux[f1_values] -= average_flux_baseline_f1

    if len(past_and_future_epochs["Extrema f2"]) != 0:

        baseline_f2 = np.concatenate((flux[f2_values][:past_and_future_epochs["Extrema f2"][0]], flux[f2_values][past_and_future_epochs["Extrema f2"][1]:]))
        average_flux_baseline_f2 = np.mean(baseline_f2)

        flux[f2_values] -= average_flux_baseline_f2

    return flux

def milky_way_extinction(ra, dec, flux, filter_wavelength):

    """ 
    Adjust for Milky Way extinction 

    Parameters:
        ra (float): Right ascension of the SN 
        dec (float): Declination of the SN 
        flux (numpy.ndarray): Differential flux of observations in microJansky
        filter_wavelength (float): Effective wavelength of the filter

    Outputs: 
        flux (numpy.ndarray): Adjusted differential flux of observations in microJansky
    """
  
    sfd = SFDQuery()
    coordinates = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))

    MW_EBV = sfd(coordinates)
    Av = 2.742 * MW_EBV

    filter_wavelength = 1.0 / (0.0001 * np.array([filter_wavelength]))

    flux = remove(fm07(filter_wavelength, Av, unit = "invum"), flux)

    return flux

def time_dilation(time, redschift):

    """ 
    Adjust for time dilation 

    Parameters:
        time (numpy.ndarray): Modified Julian Date of observations
        redschift (float): Redshift of the SN

    Outputs: 
        time (numpy.ndarray): Adjusted Modified Julian Date of observations
    """
  
    time *= 1 / (1 + redschift)

    return time 

# %%

### Data processing ###

def ztf_data_processing(ztf_names, survey_information):

    """ 
    Process raw ZTF light curve data 

    Parameters:
        ztf_names (numpy.ndarray): Internal ZTF names of the SNe to process
        survey_information (pandas.DataFrame): TNS information on the ZTF SNe

    Outputs: 
        None
    """

    f1 = "r"
    f2 = "g"

    f1_wavelength = 6366.38
    f2_wavelength = 4746.48

    ccd_threshold = pd.read_csv("../data/external/zp_thresholds_quadID.txt", comment = "#", delimiter = " |\t", header = None, engine = "python")
    ccd_threshold.columns = ["index", "g", "ignore_1", "ignore_2", "r", "ignore_3", "ignore_4", "i"]

    for name in ztf_names:

        print(name)

        # Retrieve the data
        sn_data = ztf_retrieve_data(name)

        # Retrieve supernova properties
        sn_idx = np.where(survey_information["Disc. Internal Name"] == name)

        ra = survey_information["RA"].values[sn_idx][0]
        dec = survey_information["DEC"].values[sn_idx][0]
        redschift = survey_information["Redshift"].values[sn_idx][0]
        sn_type = survey_information["Obj. Type"].values[sn_idx][0]

        # Remove noisy/bad observations
        time, flux, fluxerr, filters, zero_point, reduced_chi_squared = ztf_remove_noisy_data(ra, dec, ccd_threshold, sn_data)

        # Filter the data based on its filter
        f1_values = np.where(filters == f"ZTF_{f1}")
        f2_values = np.where(filters == f"ZTF_{f2}")

        # We only consider light curves with data in both filters 
        if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:

            # Rearange so that filters are sorted
            # Will also filter out observations with filter = i
            time = np.concatenate((time[f1_values], time[f2_values]))
            flux = np.concatenate((flux[f1_values], flux[f2_values]))
            fluxerr = np.concatenate((fluxerr[f1_values], fluxerr[f2_values]))
            filters = np.concatenate(([f1] * len(filters[f1_values]), [f2] * len(filters[f2_values])))
            zero_point = np.concatenate((zero_point[f1_values], zero_point[f2_values]))
            reduced_chi_squared = np.concatenate((reduced_chi_squared[f1_values], reduced_chi_squared[f2_values]))

            f1_values = np.where(filters == f1)
            f2_values = np.where(filters == f2)

            # Adjust the baseline to have a mean flux = 0
            past_and_future = find_baseline(flux, f1_values, f2_values)
            flux = subtract_baseline(flux, f1_values, f2_values, past_and_future)

            # Rescale flux uncertainties (if necessary)
            fluxerr = check_flux_uncertainties(fluxerr, reduced_chi_squared)

            # Apply Milky Way extraction
            flux[f1_values] = milky_way_extinction(ra, dec, flux[f1_values], f1_wavelength)
            flux[f2_values] = milky_way_extinction(ra, dec, flux[f2_values], f2_wavelength)

            # Apply time dilation
            time = time_dilation(time, redschift)

            # Only consider confident detections
            confident_detections = flux/fluxerr > 3

            time = time[confident_detections]
            flux = flux[confident_detections]
            fluxerr = fluxerr[confident_detections]
            filters = filters[confident_detections]

            f1_values = np.where(filters == f1)
            f2_values = np.where(filters == f2)

            if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:
                # Find extrema
                # Use window = 2 to make sure to not miss SN start and end
                sn_extrema = find_baseline(flux, f1_values, f2_values, 2)
                sn_extrema = np.concatenate((sn_extrema["Extrema f1"], sn_extrema["Extrema f2"]))

                time = np.concatenate((time[f1_values][sn_extrema[0] : sn_extrema[1]], time[f2_values][sn_extrema[2] : sn_extrema[3]]))
                flux = np.concatenate((flux[f1_values][sn_extrema[0] : sn_extrema[1]], flux[f2_values][sn_extrema[2] : sn_extrema[3]]))
                fluxerr = np.concatenate((fluxerr[f1_values][sn_extrema[0] : sn_extrema[1]], fluxerr[f2_values][sn_extrema[2] : sn_extrema[3]]))
                filters = np.concatenate((filters[f1_values][sn_extrema[0] : sn_extrema[1]], filters[f2_values][sn_extrema[2] : sn_extrema[3]]))

                f1_values = np.where(filters == f1)
                f2_values = np.where(filters == f2)

                if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:
                    # Delete data points 250 days before and 550 days after the peak and with a flux value below 15 microJansky
                    peak_main_idx = np.argmax(flux[f1_values])
                    peak_time = np.copy(time[peak_main_idx])
                        
                    time -= peak_time

                    to_be_deleted = np.where((time < -250) | (time > 550) | (flux < 15))

                    time = np.delete(time, to_be_deleted)
                    flux = np.delete(flux, to_be_deleted)
                    fluxerr = np.delete(fluxerr, to_be_deleted)
                    filters = np.delete(filters, to_be_deleted)

                    f1_values = np.where(filters == f1)
                    f2_values = np.where(filters == f2)
                    
                    time += peak_time

                    # We only consider light curves with enough data in both filters 
                    if len(f1_values[0]) >= 5 and len(f2_values[0]) >= 5:

                        # Save data
                        sn_data = np.column_stack((time, flux, fluxerr, filters))
                        np.save(f"../data/processed/ZTF/{name}_data.npy", sn_data)

                        # Save name to file 
                        if sn_type == "SN Ia-CSM":
                            names_file = open("../data/processed/ZTF/ZTF_SNe_Ia_CSM.txt", "a")
                            names_file.write(name + "\n")
                            names_file.close()
                        
                        elif sn_type == "SN IIn":
                            names_file = open("../data/processed/ZTF/ZTF_SNe_IIn.txt", "a")
                            names_file.write(name + "\n")
                            names_file.close()

                        # Save plot 
                        ztf_plot_data(name, time, flux, fluxerr, filters, save_fig = True)

def atlas_data_processing(atlas_names, survey_information):

    """ 
    Process raw ATLAS light curve data 

    Parameters:
        atlas_names (numpy.ndarray): TNS names of the ATLAS SNe to process
        survey_information (pandas.DataFrame): TNS information on the ATLAS SNe

    Outputs: 
        None
    """

    f1 = "o"
    f2 = "c"

    f1_wavelength = 6629.83
    f2_wavelength = 5182.42

    for name in atlas_names:

        print(name)
            
        # Retrieve the data
        time, flux, fluxerr, filters = atlas_retrieve_data(name)

        # Retrieve supernova properties
        sn_idx = np.where(survey_information["Name"] == name.replace("_", " "))

        internal_name = survey_information["Disc. Internal Name"].values[sn_idx][0]
        ra = survey_information["RA"].values[sn_idx][0]
        dec = survey_information["DEC"].values[sn_idx][0]
        redshift = survey_information["Redshift"].values[sn_idx][0]
        sn_type = survey_information["Obj. Type"].values[sn_idx][0]

        # Remove noisy/bad observations
        time, flux, fluxerr, filters = atlas_remove_noisy_data(time, flux, fluxerr, filters)

        # Filter the data based on the used filter
        f1_values = np.where(filters == f1)
        f2_values = np.where(filters == f2)

        # We only consider light curves with data in both filters 
        if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:

            # Rearange so that filters are sorted
            time = np.concatenate((time[f1_values], time[f2_values]))
            flux = np.concatenate((flux[f1_values], flux[f2_values]))
            fluxerr = np.concatenate((fluxerr[f1_values], fluxerr[f2_values]))
            filters = np.concatenate(([f1] * len(filters[f1_values]), [f2] * len(filters[f2_values])))

            f1_values = np.where(filters == f1)
            f2_values = np.where(filters == f2)

            # Adjust the baseline so that its mean flux = 0
            past_and_future = find_baseline(flux, f1_values, f2_values)
            flux = subtract_baseline(flux, f1_values, f2_values, past_and_future)

            # Apply Milky Way extraction
            flux[f1_values] = milky_way_extinction(ra, dec, flux[f1_values], f1_wavelength)
            flux[f2_values] = milky_way_extinction(ra, dec, flux[f2_values], f2_wavelength)

            # Apply time dilation
            time = time_dilation(time, redshift)

            # Only consider confident detections
            confident_detections = flux/fluxerr > 3

            time = time[confident_detections]
            flux = flux[confident_detections]
            fluxerr = fluxerr[confident_detections]
            filters = filters[confident_detections]

            f1_values = np.where(filters == f1)
            f2_values = np.where(filters == f2)

            if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:
                # Find extrema
                sn_extrema = find_baseline(flux, f1_values, f2_values, 2)
                sn_extrema = np.concatenate((sn_extrema["Extrema f1"], sn_extrema["Extrema f2"]))

                time = np.concatenate((time[f1_values][sn_extrema[0] : sn_extrema[1]], time[f2_values][sn_extrema[2] : sn_extrema[3]]))
                flux = np.concatenate((flux[f1_values][sn_extrema[0] : sn_extrema[1]], flux[f2_values][sn_extrema[2] : sn_extrema[3]]))
                fluxerr = np.concatenate((fluxerr[f1_values][sn_extrema[0] : sn_extrema[1]], fluxerr[f2_values][sn_extrema[2] : sn_extrema[3]]))
                filters = np.concatenate((filters[f1_values][sn_extrema[0] : sn_extrema[1]], filters[f2_values][sn_extrema[2] : sn_extrema[3]]))

                f1_values = np.where(filters == f1)
                f2_values = np.where(filters == f2)

                if len(f1_values[0]) != 0 and len(f2_values[0]) != 0:
                    # Delete datapoint 250 days before and 550 days after the peak and with a flux value below 15 microJansky
                    peak_main_idx = np.argmax(flux[f1_values])
                    peak_time = np.copy(time[peak_main_idx])
                        
                    time -= peak_time

                    to_be_deleted = np.where((time < -250) | (time > 550) | (flux < 15))

                    time = np.delete(time, to_be_deleted)
                    flux = np.delete(flux, to_be_deleted)
                    fluxerr = np.delete(fluxerr, to_be_deleted)
                    filters = np.delete(filters, to_be_deleted)

                    f1_values = np.where(filters == f1)
                    f2_values = np.where(filters == f2)
                    
                    time += peak_time

                    # We only consider light curves with data in both filters 
                    if len(f1_values[0]) >= 5 and len(f2_values[0]) >= 5:
                        # Save data
                        sn_data = np.column_stack((time, flux, fluxerr, filters))
                        np.save(f"../data/processed/ATLAS/{internal_name}_data.npy", sn_data)

                        # Save name to file 
                        if sn_type == "SN Ia-CSM":
                            names_file = open("../data/processed/ATLAS/ATLAS_SNe_Ia_CSM.txt", "a")
                            names_file.write(internal_name + "\n")
                            names_file.close()
                        
                        elif sn_type == "SN IIn":
                            names_file = open("../data/processed/ATLAS/ATLAS_SNe_IIn.txt", "a")
                            names_file.write(internal_name + "\n")
                            names_file.close()

                        # Save plot 
                        atlas_plot_data(internal_name, time, flux, fluxerr, filters, save_fig = True)
                
# %%
    
if __name__ == '__main__':

    # ZTF
    ztf_information = pd.read_csv("../data/external/ZTF_info.csv")
    ztf_names = ztf_information["Disc. Internal Name"][~pd.isnull(ztf_information["Disc. Internal Name"])].values

    ztf_data_processing(ztf_names, ztf_information)

    #ATLAS
    atlas_information = pd.read_csv("../data/external/ATLAS_info.csv")
    atlas_names = atlas_information["Name"].values
    atlas_names = [name.replace(" ", "_") for name in atlas_names]

    atlas_data_processing(atlas_names, atlas_information)

# %%
