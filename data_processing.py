# %% 

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from dustmaps.sfd import SFDQuery
from extinction import fm07, remove

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True

# %%

### ZTF ###

# Generate data
def ztf_retrieve_data(ztf_name):

    with open(f"Data/ZTF_forced_photometry_data/raw/{ztf_name}.txt") as f:

        # Ignore lines with parameter explenation
        lines = (line for line in f if not line.startswith('#'))

        # Ignore line with parameter names
        ztf_data = np.genfromtxt(lines, skip_header = 1, missing_values = "null", dtype = "str")

    # Supernova data
    time = np.array(ztf_data[:, 22])
    flux = np.array(ztf_data[:, 24])
    fluxerr = np.array(ztf_data[:, 25])
    filters = np.array(ztf_data[:, 4])

    # Parameters used to identify noisy/bad data
    infobitssci = np.array(ztf_data[:, 6]).astype(np.float32)
    scisigpix = np.array(ztf_data[:, 9]).astype(np.float32)
    sciinpseeing = np.array(ztf_data[:, 7]).astype(np.float32)
    status = np.array(ztf_data[:, -1]).astype(str)

    ccd = np.array(ztf_data[:, 2])
    ccd_amplifier = np.array(ztf_data[:, 3])
    ncalmatches = np.array(ztf_data[:, 15])

    zero_point = np.array(ztf_data[:, 10])
    zero_point_rms = np.array(ztf_data[:, 12])
    reduced_chi_squared = np.array(ztf_data[:, 27])

    return time, flux, fluxerr, filters, infobitssci, scisigpix, \
           sciinpseeing, status, ccd, ccd_amplifier, ncalmatches, \
           zero_point, zero_point_rms, reduced_chi_squared

# Calulate airmass for quality cuts
def calculate_airmass(ra, dec, time):

    palomar = EarthLocation.of_site('Palomar')

    target = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))
    time = Time(time, format = "jd")

    target_in_altaz = target.transform_to(AltAz(obstime = time, location = palomar))
    airmass = target_in_altaz.secz.value

    return airmass

# Remove noisy/bad observations
def ztf_remove_noisy_data(ra, dec, ccd_threshold, time, flux, fluxerr, filters, infobitssci, scisigpix, sciinpseeing, status, ccd, ccd_amplifier, ncalmatches, zero_point, zero_point_rms, reduced_chi_squared):

    # Filter out bad processing epochs
    bad_observations = np.where(((status != "0") & (status != "56") & (status != "57") & (status != "62") & (status != "65")))[0]

    # Filter out data with missing flux measurements
    missing_data = np.where(flux == "null")[0]

    # Filter out data of a bad quality (possibly contaminated by clouds or the moon)
    bad_infobitssci = np.where(infobitssci > 0)[0]
    bad_scisigpix = np.where(scisigpix > 25)[0]
    bad_sciinpseeing = np.where(sciinpseeing > 4)[0]

    # Delte bad data 
    indices_to_delete = np.sort(np.unique(np.concatenate((bad_observations, missing_data, bad_infobitssci, bad_scisigpix, bad_sciinpseeing))))

    time = np.delete(time, indices_to_delete).astype(np.float32)
    flux = np.delete(flux, indices_to_delete).astype(np.float32)
    fluxerr = np.delete(fluxerr, indices_to_delete).astype(np.float32)
    filters = np.delete(filters, indices_to_delete)

    ccd = np.delete(ccd, indices_to_delete).astype(np.float32)
    ccd_amplifier = np.delete(ccd_amplifier, indices_to_delete).astype(np.float32)
    ncalmatches = np.delete(ncalmatches, indices_to_delete).astype(np.float32)
    zero_point = np.delete(zero_point, indices_to_delete).astype(np.float32)
    zero_point_rms = np.delete(zero_point_rms, indices_to_delete).astype(np.float32)
    reduced_chi_squared = np.delete(reduced_chi_squared, indices_to_delete).astype(np.float32)

    # Quality cuts
    airmass = calculate_airmass(ra, dec, time)
    rcid = 4 * (ccd - 1) + ccd_amplifier - 1

    quality_cuts_f1 = np.where((filters == "ZTF_r") & ((zero_point > (26.65 - 0.15 * airmass)) | (zero_point_rms > 0.05) | (ncalmatches < 120) | (zero_point < (ccd_threshold["r"].iloc[rcid].to_numpy() - 0.15 * airmass))))[0]
    quality_cuts_f2 = np.where((filters == "ZTF_g") & ((zero_point > (26.7 - 0.2 * airmass)) | (zero_point_rms > 0.06) | (ncalmatches < 80) | (zero_point < (ccd_threshold["g"].iloc[rcid].to_numpy() - 0.2 * airmass))))[0]

    indices_to_delete = np.sort(np.unique(np.concatenate((quality_cuts_f1, quality_cuts_f2))))

    time = np.delete(time, indices_to_delete) -  2400000.5
    flux = np.delete(flux, indices_to_delete) / 10
    fluxerr = np.delete(fluxerr, indices_to_delete) / 10
    filters = np.delete(filters, indices_to_delete)

    zero_point = np.delete(zero_point, indices_to_delete)
    reduced_chi_squared = np.delete(reduced_chi_squared, indices_to_delete)

    return time, flux, fluxerr, filters, \
           zero_point, reduced_chi_squared

# Check if the flux uncertainties need to be rescaled/are good estimations
def check_flux_uncertainties(fluxerr, reduced_chi_squared):

    average_reduced_chi_squared = np.mean(reduced_chi_squared)

    if not np.isclose(average_reduced_chi_squared, 1, 0.5):

        fluxerr *= np.sqrt(average_reduced_chi_squared)

    return fluxerr

# Conversion
def ztf_micro_flux_to_magnitude(flux, fluxerr, zero_point):

    mag = np.empty(len(flux))
    magerr = np.empty(len(flux))

    confident_detections = flux / fluxerr > 3

    # Confident detections
    mag[confident_detections] = zero_point[confident_detections] - 2.5 * np.log10(10 * flux[confident_detections])
    magerr[confident_detections] = 1.0857 * fluxerr[confident_detections] / flux[confident_detections]

    # Non-detections
    mag[~confident_detections] = zero_point[~confident_detections] - 2.5 * np.log10(10 * 5 * fluxerr[~confident_detections])
    magerr[~confident_detections] = 0

    return mag, magerr, confident_detections

# Load data saved in directory
def ztf_load_data(ztf_name):

    ztf_data = np.load(f"Data/ZTF_forced_photometry_data/processed/{ztf_name}_data.npy")

    time = ztf_data[:, 0]
    flux = ztf_data[:, 1]
    fluxerr = ztf_data[:, 2]
    filters = ztf_data[:, 3]

    return np.array(time).astype(np.float32), np.array(flux).astype(np.float32), \
           np.array(fluxerr).astype(np.float32), np.array(filters)

# Plot data
def ztf_plot_data(ztf_name, time, flux, fluxerr, filters, extrema, save_fig = False):

    r_values = np.where(filters == "r")
    g_values = np.where(filters == "g")

    if time[r_values][extrema[0]] < time[g_values][extrema[2]]:
        time_first_detection = time[r_values][extrema[0]]
        time -= time_first_detection

    else: 
        time_first_detection = time[g_values][extrema[2]]
        time -= time_first_detection
    
    plt.errorbar(time[r_values], flux[r_values], yerr = fluxerr[r_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = "Band: r")
    plt.errorbar(time[g_values], flux[g_values], yerr = fluxerr[g_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = "Band: g")

    plt.xlabel("Days since first detection", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {ztf_name}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/ZTF_lightcurves/Light_curve_{ztf_name}", dpi = 300)
        plt.close()
    else:
        plt.show()

# %%

### ATLAS ###

# Generate data
def atlas_retrieve_data(atlas_name):

    # Load stacked and cleaned data 
    atlas_data =  np.loadtxt(f"Data/ATLAS_forced_photometry_data/cleaned_and_stacked/{atlas_name}_atlas_fp_stacked_2_days.txt", delimiter = ",", dtype = str)

    time = atlas_data[1:, 0].astype(np.float32)
    flux = atlas_data[1:, 1].astype(np.float32)
    fluxerr = atlas_data[1:, 2].astype(np.float32)
    filters = atlas_data[1:, 3].astype(str)

    return time, flux, fluxerr, filters

# Remove noisy/bad observations
def atlas_remove_noisy_data(time, flux, fluxerr, filters):

    # Identify noisy observations
    delete_flux_error = np.where(fluxerr > 40)

    time = np.delete(time, delete_flux_error)
    flux = np.delete(flux, delete_flux_error)
    fluxerr = np.delete(fluxerr, delete_flux_error)
    filters = np.delete(filters, delete_flux_error)

    return time, flux, fluxerr, filters

# Conversion
def atlas_micro_flux_to_magnitude(flux, fluxerr):

    magnitude = -2.5 * np.log10(flux) + 23.9
    magnitude_error = np.abs(-2.5 * (fluxerr / (flux * np.log(10))))

    return magnitude, magnitude_error

# Load data saved in directory
def atlas_load_data(atlas_name):

    atlas_data = np.load(f"Data/ATLAS_forced_photometry_data/processed/{atlas_name}_data.npy")

    time = atlas_data[:, 0]
    flux = atlas_data[:, 1]
    fluxerr = atlas_data[:, 2]
    filters = atlas_data[:, 3]

    return np.array(time).astype(np.float32), np.array(flux).astype(np.float32), \
           np.array(fluxerr).astype(np.float32), np.array(filters)


# Plot data
def atlas_plot_data(atlas_name, time, flux, fluxerr, filters, extrema, save_fig = False):
    
    o_values = np.where(filters == "o")
    c_values = np.where(filters == "c")

    if time[o_values][extrema[0]] < time[c_values][extrema[2]]:
        time_first_detection = time[o_values][extrema[0]]
        time -= time_first_detection

    else: 
        time_first_detection = time[c_values][extrema[2]]
        time -= time_first_detection

    plt.errorbar(time[o_values], flux[o_values], yerr = fluxerr[o_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:blue", label = "Band: o")
    plt.errorbar(time[c_values], flux[c_values], yerr = fluxerr[c_values], fmt = "o", markersize = 4, capsize = 2, color = "tab:orange", label = "Band: c")

    plt.xlabel("Days since first detection", fontsize = 13)
    plt.ylabel("Flux $(\mu Jy)$", fontsize = 13)
    plt.title(f"Light curve of SN {atlas_name}.")
    plt.grid(alpha = 0.3)
    plt.legend()
    if save_fig:
        plt.savefig(f"Plots/ATLAS_lightcurves/Light_curve_{atlas_name}", dpi = 300)
        plt.close()
    else:
        plt.show()

# %%

### General ###

# Baseline subtraction
    # Has to be after cleaning because the peak needs to be determined
    # Often highest value has large error and is probably background noise

def find_baseline(flux, filter_f1, filter_f2, window = 5):

    past_and_future_epochs =  {"Extrema f1":[], "Extrema f2":[]}

    for filter_number, filter_idx in enumerate([filter_f1, filter_f2]):
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
    for filter_number, filter_idx in enumerate([filter_f1, filter_f2]):

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

def subtract_baseline(flux, filter_f1, filter_f2, past_and_future_epochs):

    if len(past_and_future_epochs["Extrema f1"]) != 0:

        baseline_f1 = np.concatenate((flux[filter_f1][:past_and_future_epochs["Extrema f1"][0]], flux[filter_f1][past_and_future_epochs["Extrema f1"][1]:]))
        average_flux_baseline_f1 = np.mean(baseline_f1)

        flux[filter_f1] -= average_flux_baseline_f1

    if len(past_and_future_epochs["Extrema f2"]) != 0:

        baseline_f2 = np.concatenate((flux[filter_f2][:past_and_future_epochs["Extrema f2"][0]], flux[filter_f2][past_and_future_epochs["Extrema f2"][1]:]))
        average_flux_baseline_f2 = np.mean(baseline_f2)

        flux[filter_f2] -= average_flux_baseline_f2

    return flux

# Return the flux corrected for Milky Way extinction
def milky_way_extinction(ra, dec, flux, filter_wavelength):
    
    sfd = SFDQuery()
    coordinates = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))

    MW_EBV = sfd(coordinates)
    Av = 2.742 * MW_EBV

    filter_wavelength = 1.0 / (0.0001 * np.array([filter_wavelength]))

    flux = remove(fm07(filter_wavelength, Av, unit = "invum"), flux)

    return flux

# Return the time corrected for time dilation
def time_dilation(time, redschift):

    time *= 1 / (1 + redschift)

    return time 

# %%

### Data processing ###

def ztf_data_processing(ztf_names, survey_information):

    f1 = "r"
    f2 = "g"

    f1_wavelength = 6366.38
    f2_wavelength = 4746.48

    ccd_threshold = pd.read_csv("Data/zp_thresholds_quadID.txt", comment = "#", delimiter = " |\t", header = None, engine = "python")
    ccd_threshold.columns = ["index", "g", "ignore_1", "ignore_2", "r", "ignore_3", "ignore_4", "i"]

    for name in ztf_names:

        print(name)

        # Retrieve the data
        time, flux, fluxerr, filters, infobitssci, scisigpix, sciinpseeing, status, ccd, ccd_amplifier, ncalmatches, zero_point, zero_point_rms, reduced_chi_squared = ztf_retrieve_data(name)

        # Retrieve supernova properties
        SN_idx = np.where(survey_information["Disc. Internal Name"] == name)

        ra = survey_information["RA"].values[SN_idx][0]
        dec = survey_information["DEC"].values[SN_idx][0]
        redschift = survey_information["Redshift"].values[SN_idx][0]
        SN_type = survey_information["Obj. Type"].values[SN_idx][0]

        # Remove noisy/bad observations
        time, flux, fluxerr, filters, zero_point, reduced_chi_squared = ztf_remove_noisy_data(ra, dec, ccd_threshold, time, flux, fluxerr, filters, infobitssci, scisigpix, sciinpseeing, status, ccd, ccd_amplifier, ncalmatches, zero_point, zero_point_rms, reduced_chi_squared)

        # Filter the data based on the used filter
        filter_f1 = np.where(filters == f"ZTF_{f1}")
        filter_f2 = np.where(filters == f"ZTF_{f2}")

        # We only consider light curves with data in both filters 
        if len(filter_f1[0]) != 0 and len(filter_f2[0]) != 0:

            # Rearange so that filters are sorted
            # Will also filter out observations with filter = i
            time = np.concatenate((time[filter_f1], time[filter_f2]))
            flux = np.concatenate((flux[filter_f1], flux[filter_f2]))
            fluxerr = np.concatenate((fluxerr[filter_f1], fluxerr[filter_f2]))
            filters = np.concatenate(([f1] * len(filters[filter_f1]), [f2] * len(filters[filter_f2])))
            zero_point = np.concatenate((zero_point[filter_f1], zero_point[filter_f2]))
            reduced_chi_squared = np.concatenate((reduced_chi_squared[filter_f1], reduced_chi_squared[filter_f2]))

            filter_f1 = np.where(filters == f1)
            filter_f2 = np.where(filters == f2)

            # Adjust the baseline to flux = 0
            past_and_future = find_baseline(flux, filter_f1, filter_f2)
            flux = subtract_baseline(flux, filter_f1, filter_f2, past_and_future)

            # Adjust estimates of flux uncertainties (if necessary)
            fluxerr = check_flux_uncertainties(fluxerr, reduced_chi_squared)

            # Apply Milky Way extraction
            flux[filter_f1] = milky_way_extinction(ra, dec, flux[filter_f1], f1_wavelength)
            flux[filter_f2] = milky_way_extinction(ra, dec, flux[filter_f2], f2_wavelength)

            # Apply time dilation
            time = time_dilation(time, redschift)

            # Only consider confident detections
            confident_detections = flux/fluxerr > 3

            time = time[confident_detections]
            flux = flux[confident_detections]
            fluxerr = fluxerr[confident_detections]
            filters = filters[confident_detections]

            filter_f1 = np.where(filters == f1)
            filter_f2 = np.where(filters == f2)

            if len(filter_f1[0]) != 0 and len(filter_f2[0]) != 0:
                # Find extrema
                SN_extrema = find_baseline(flux, filter_f1, filter_f2, 2)
                SN_extrema = np.concatenate((SN_extrema["Extrema f1"], SN_extrema["Extrema f2"]))

                time = np.concatenate((time[filter_f1][SN_extrema[0] : SN_extrema[1]], time[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                flux = np.concatenate((flux[filter_f1][SN_extrema[0] : SN_extrema[1]], flux[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                fluxerr = np.concatenate((fluxerr[filter_f1][SN_extrema[0] : SN_extrema[1]], fluxerr[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                filters = np.concatenate((filters[filter_f1][SN_extrema[0] : SN_extrema[1]], filters[filter_f2][SN_extrema[2] : SN_extrema[3]]))

                # Delete datapoint 250 days before and 550 days after the peak and with a flux value below 15 microJansky
                to_be_deleted = np.where((time < -250) | (time > 550) | (flux < 15))

                time = np.delete(time, to_be_deleted)
                flux = np.delete(flux, to_be_deleted)
                fluxerr = np.delete(fluxerr, to_be_deleted)
                filters = np.delete(filters, to_be_deleted)

                filter_f1 = np.where(filters == f1)
                filter_f2 = np.where(filters == f2)

                # We only consider light curves with data in both filters 
                if len(filter_f1[0]) >= 5 and len(filter_f2[0]) >= 5:

                    # Save data
                    SN_data = np.column_stack((time, flux, fluxerr, filters))
                    np.save(f"Data/ZTF_forced_photometry_data/processed/{name}_data.npy", SN_data)

                    # Save name to file 
                    if SN_type == "SN Ia-CSM":
                        names_file = open("Data/ZTF_SNe_Ia_CSM.txt", "a")
                        names_file.write(name + "\n")
                        names_file.close()
                    
                    elif SN_type == "SN IIn":
                        names_file = open("Data/ZTF_SNe_IIn.txt", "a")
                        names_file.write(name + "\n")
                        names_file.close()

                    # Save plot 
                    ztf_plot_data(name, time, flux, fluxerr, filters, SN_extrema, save_fig = True)

def atlas_data_processing(atlas_names, survey_information):

    f1 = "o"
    f2 = "c"

    f1_wavelength = 6629.83
    f2_wavelength = 5182.42

    for name in atlas_names:

        print(name)
            
        # Retrieve the data
        time, flux, fluxerr, filters = atlas_retrieve_data(name)

        # Retrieve supernova properties
        SN_idx = np.where(survey_information["Name"] == name.replace("_", " "))

        internal_name = survey_information["Disc. Internal Name"].values[SN_idx][0]
        ra = survey_information["RA"].values[SN_idx][0]
        dec = survey_information["DEC"].values[SN_idx][0]
        redshift = survey_information["Redshift"].values[SN_idx][0]
        SN_type = survey_information["Obj. Type"].values[SN_idx][0]

        # Remove noisy/bad observations
        time, flux, fluxerr, filters = atlas_remove_noisy_data(time, flux, fluxerr, filters)

        # Filter the data based on the used filter
        filter_f1 = np.where(filters == f1)
        filter_f2 = np.where(filters == f2)

        # We only consider light curves with data in both filters 
        if len(filter_f1[0]) != 0 and len(filter_f2[0]) != 0:

            # Rearange so that filters are sorted
            # Will also filter out observations with filter = i
            time = np.concatenate((time[filter_f1], time[filter_f2]))
            flux = np.concatenate((flux[filter_f1], flux[filter_f2]))
            fluxerr = np.concatenate((fluxerr[filter_f1], fluxerr[filter_f2]))
            filters = np.concatenate(([f1] * len(filters[filter_f1]), [f2] * len(filters[filter_f2])))

            filter_f1 = np.where(filters == f1)
            filter_f2 = np.where(filters == f2)

            # Adjust the baseline to flux = 0
            past_and_future = find_baseline(flux, filter_f1, filter_f2)
            flux = subtract_baseline(flux, filter_f1, filter_f2, past_and_future)

            # Apply Milky Way extraction
            flux[filter_f1] = milky_way_extinction(ra, dec, flux[filter_f1], f1_wavelength)
            flux[filter_f2] = milky_way_extinction(ra, dec, flux[filter_f2], f2_wavelength)

            # Apply time dilation
            time = time_dilation(time, redshift)

            # Only consider confident detections
            confident_detections = flux/fluxerr > 3

            time = time[confident_detections]
            flux = flux[confident_detections]
            fluxerr = fluxerr[confident_detections]
            filters = filters[confident_detections]

            filter_f1 = np.where(filters == f1)
            filter_f2 = np.where(filters == f2)

            if len(filter_f1[0]) != 0 and len(filter_f2[0]) != 0:
                # Find extrema
                SN_extrema = find_baseline(flux, filter_f1, filter_f2, 2)
                SN_extrema = np.concatenate((SN_extrema["Extrema f1"], SN_extrema["Extrema f2"]))

                time = np.concatenate((time[filter_f1][SN_extrema[0] : SN_extrema[1]], time[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                flux = np.concatenate((flux[filter_f1][SN_extrema[0] : SN_extrema[1]], flux[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                fluxerr = np.concatenate((fluxerr[filter_f1][SN_extrema[0] : SN_extrema[1]], fluxerr[filter_f2][SN_extrema[2] : SN_extrema[3]]))
                filters = np.concatenate((filters[filter_f1][SN_extrema[0] : SN_extrema[1]], filters[filter_f2][SN_extrema[2] : SN_extrema[3]]))

                # Delete datapoint 250 days before and 550 days after the peak and with a flux value below 15 microJansky
                to_be_deleted = np.where((time < -250) | (time > 550) | (flux < 15))

                time = np.delete(time, to_be_deleted)
                flux = np.delete(flux, to_be_deleted)
                fluxerr = np.delete(fluxerr, to_be_deleted)
                filters = np.delete(filters, to_be_deleted)

                filter_f1 = np.where(filters == f1)
                filter_f2 = np.where(filters == f2)

                # We only consider light curves with data in both filters 
                if len(filter_f1[0]) >= 5 and len(filter_f2[0]) >= 5:
                    # Save data
                    SN_data = np.column_stack((time, flux, fluxerr, filters))
                    np.save(f"Data/ATLAS_forced_photometry_data/processed/{internal_name}_data.npy", SN_data)

                    # Save name to file 
                    if SN_type == "SN Ia-CSM":
                        names_file = open("Data/ATLAS_SNe_Ia_CSM.txt", "a")
                        names_file.write(internal_name + "\n")
                        names_file.close()
                    
                    elif SN_type == "SN IIn":
                        names_file = open("Data/ATLAS_SNe_IIn.txt", "a")
                        names_file.write(internal_name + "\n")
                        names_file.close()

                    # Save plot 
                    atlas_plot_data(internal_name, time, flux, fluxerr, filters, SN_extrema, save_fig = True)
            
# %%
    
if __name__ == '__main__':

    # ZTF
    ztf_information = pd.read_csv("Data/ZTF_info.csv")
    ztf_names = ztf_information["Disc. Internal Name"][~pd.isnull(ztf_information["Disc. Internal Name"])].values
    
    ztf_data_processing(ztf_names, ztf_information)

    #ATLAS
    atlas_information = pd.read_csv("Data/ATLAS_info.csv")
    atlas_names = atlas_information["Name"].values
    atlas_names = [name.replace(" ", "_") for name in atlas_names]

    atlas_data_processing(atlas_names, atlas_information)

# %%
