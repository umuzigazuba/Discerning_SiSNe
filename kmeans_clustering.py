# %%

from data_processing import ztf_load_data, atlas_load_data, atlas_micro_flux_to_magnitude
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True
plt.rcParams['axes.axisbelow'] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

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
        print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.25].iloc[0]).dropna())
        print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
        print("\n******************************************************************")

    plt.figure(figsize = (8, 6))

    unique_labels = np.unique(SN_type)
    for i, lbl in enumerate(unique_labels):

        class_data = pca_df[pca_df["SN_type"] == lbl]
        plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = list(colours.values())[i], label = lbl)

    plt.title(f"PCA plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.show()

def plot_PCA_with_clusters(parameter_values, SN_type, kmeans, best_number, number_of_peaks, save_fig = False):

    pca = decomposition.PCA(n_components = 2, random_state = 2804)
    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data[:, :2], columns = ["Dimension 1", "Dimension 2"])
    pca_df["SN_type"] = SN_type
    pca_df["number_of_peaks"] = number_of_peaks
    # pca_df["cluster"] = kmeans.labels_

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

    unique_labels = np.unique(SN_type)

    # Create proxy artists for the background clusters
    cluster_colours = {"green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE"}
    clusters = [mpatches.Patch(color = list(cluster_colours.values())[idx], label = f"K-means cluster {idx + 1}") for idx in range(best_number)]

    # Plot the background regions (colored by KMeans predictions)
    plt.figure(figsize = (8, 6))
    plt.contourf(xx, yy, grid_clusters, levels = [-0.5, 0.5, 1.5, 2.5, 3.5], colors = list(cluster_colours.values()), alpha = 0.3)

    for i, lbl in enumerate(unique_labels):
        # Filter the data for the current label
        class_data = pca_df[pca_df["SN_type"] == lbl]

        # Plot circles for number_of_peaks == 1 and triangles for number_of_peaks == 2
        one_peak_data = class_data[class_data["number_of_peaks"] == 1]
        two_peak_data = class_data[class_data["number_of_peaks"] == 2]

        # Plot one-peak supernovae (circles)
        if len(one_peak_data) != 0:
            plt.scatter(one_peak_data["Dimension 1"], one_peak_data["Dimension 2"], s = 65,
                        c = list(colours.values())[i], marker = 'o', label = f'{lbl} (one peak)', edgecolor = 'k')

        # Plot two-peak supernovae (triangles)
        if len(two_peak_data) != 0:
            plt.scatter(two_peak_data["Dimension 1"], two_peak_data["Dimension 2"], s = 65,
                        c = list(colours.values())[i], marker = '^', label = f'{lbl} (two peaks)', edgecolor = 'k')

    # Labels and legend
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.3)
    plt.legend()
    plt.legend(handles = plt.legend().legendHandles + clusters)
    if type(save_fig) == str:
        name = save_fig.replace("_", " ")
        plt.title(f"K-means clusters of the {name}.")
        plt.savefig(f"Plots/Results/PCA_plot_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"PCA plot")
        plt.show()
  
def dimensionality_reduction(parameter_values, new_dimensions):

    pca = decomposition.PCA(n_components = new_dimensions, random_state = 2804)

    low_dimension_parameter_values = pca.fit_transform(parameter_values)
    # print("Cumulative variance explained by 2 principal components: {:.2%}".format(np.sum(pca.explained_variance_ratio_)))

    return low_dimension_parameter_values

# %%

def plot_SN_collection(survey, fitting_parameters, number_of_peaks):

    collection_times_f1 = []
    collection_fluxes_f1 = []

    collection_times_f2 = []
    collection_fluxes_f2 = []

    for idx in range(len(fitting_parameters)):
    
        if survey == "ZTF":
            f1 = "r"
            f2 = "g"

            time, flux, _, filters = ztf_load_data(fitting_parameters[idx, 0])

        if survey == "ATLAS":
            f1 = "o"
            f2 = "c"

            # Load the data
            time, flux, _, filters = atlas_load_data(fitting_parameters[idx, 0])

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

def number_of_clusters(parameters, save_fig = False):

    n_clusters = np.array([2, 3, 4])
    silhouette_scores = []

    parameter_grid = ParameterGrid({"n_clusters": n_clusters})
    best_score = -1

    kmeans_model = KMeans(random_state = 2804)

    for p in parameter_grid:
        
        # Set number of clusters
        kmeans_model.set_params(**p)    
        kmeans_model.fit(parameters)

        ss = metrics.silhouette_score(parameters, kmeans_model.labels_)
        silhouette_scores += [ss] 
        if ss > best_score:
            best_score = ss

    mask = np.where(silhouette_scores == best_score)

    # plotting silhouette score
    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align = "center", color = colours["blue"], width = 0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    if type(save_fig) == str:
        name = save_fig.replace("_", " ")
        plt.title(f"Silhouette score of the {name}.")
        plt.grid(alpha = 0.5)
        plt.savefig(f"Plots/Results/silhouette_score_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"Silhouette score. Best number of clusters = {n_clusters[mask][0]}. Score = {round(best_score, 2)}.")
        plt.grid(alpha = 0.5)
        plt.show()

    return n_clusters[mask][0]

def loss_of_information(parameter_values, percentage = 75, save_fig = False):

    cummulative_variance = []

    possible_dimensions = np.arange(2, len(parameter_values[0]) + 1)

    for dimension in possible_dimensions:

        pca = decomposition.PCA(n_components = dimension, random_state = 2804)
        pca.fit(parameter_values)
        cummulative_variance.append(np.sum(pca.explained_variance_ratio_))

    plt.bar(range(len(cummulative_variance)), np.array(cummulative_variance) * 100, align = "center", color = colours["blue"], width = 0.5)
    plt.xticks(range(len(cummulative_variance)), list(possible_dimensions))
    plt.axhline(percentage, linestyle = "dashed", linewidth = 3, color = "black")
    plt.xlabel("Number of dimensions")
    plt.ylabel("Cummulative variance (\%)")
    if type(save_fig) == str:
        name = save_fig.replace("_", " ")
        plt.title(f"Cumulative variance of the {name}.")
        plt.grid(alpha = 0.3)
        plt.savefig(f"Plots/Results/cummulative_variance_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"Cumulative variance.")
        plt.grid(alpha = 0.3)
        plt.show()

    best_number_idx = np.argmin(np.abs(np.array(cummulative_variance) - percentage / 100))
    best_number = possible_dimensions[best_number_idx]

    return best_number
    
# %%

if __name__ == '__main__':

    survey = "ZTF"
    
    fitting_parameters = np.load(f"Data/Input_ML/{survey}/fitting_parameters.npy", allow_pickle = True)
    fitting_parameters_one_peak = np.load(f"Data/Input_ML/{survey}/fitting_parameters_one_peak.npy", allow_pickle = True)
    global_parameters = np.load(f"Data/Input_ML/{survey}/global_parameters.npy")
    global_parameters_one_peak = np.load(f"Data/Input_ML/{survey}/global_parameters_one_peak.npy")
    number_of_peaks = np.load(f"Data/Input_ML/{survey}/number_of_peaks.npy")
    SN_labels = np.load(f"Data/Input_ML/{survey}/SN_labels.npy")
    SN_labels_color = np.load(f"Data/Input_ML/{survey}/SN_labels_color.npy")
    scaler = StandardScaler()

    # %%

    # #### No redshift

    # fitting_parameters_names = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
    #                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"]

    # fitting_parameters_scaled = scaler.fit_transform(fitting_parameters[:, 1:15])

    # # plot_PCA(fitting_parameters_scaled, SN_labels, fitting_parameters_names)
    # best_number = number_of_clusters(fitting_parameters_scaled)

    # kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    # kmeans.fit(fitting_parameters_scaled)

    # plot_PCA_with_clusters(fitting_parameters_scaled, SN_labels, kmeans, best_number, number_of_peaks)

    # %%

    # #### With redshift

    # redshifts = retrieve_redshift(fitting_parameters[:, 0], survey)

    # peak_abs_magnitude = []
    # for idx in range(len(redshifts)):

    #     peak_abs_magnitude.append(calculate_peak_absolute_magnitude(global_parameters[idx, 0], redshifts[idx]))

    # fitting_parameters_names = ["A_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
    #                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2", \
    #                             "M_r", "z"]

    # fitting_parameters_redshift = np.concatenate((fitting_parameters[:, 1].reshape(len(SN_labels), 1), fitting_parameters[:, 3:15], np.array(peak_abs_magnitude).reshape(len(SN_labels), 1), np.array(redshifts).reshape(len(SN_labels), 1)), axis = 1)

    # fitting_parameters_scaled = scaler.fit_transform(fitting_parameters_redshift)

    # plot_PCA(fitting_parameters_scaled, SN_labels, fitting_parameters_names)
    # best_number = number_of_clusters(fitting_parameters_scaled)

    # kmeans = KMeans(n_clusters = best_number)
    # kmeans.fit(fitting_parameters_scaled)

    # plot_PCA_with_clusters(fitting_parameters_scaled, SN_labels, kmeans, number_of_peaks)


    # %%
    #### model parameteres one and double peak

    parameters_two_peaks_names = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                                "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"] \
                                + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]

    parameters_two_peaks_scaled = scaler.fit_transform(fitting_parameters[:, 1:])

    plot_PCA(parameters_two_peaks_scaled, SN_labels, parameters_two_peaks_names)
    best_number = number_of_clusters(parameters_two_peaks_scaled)
    # best_number = 2

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(parameters_two_peaks_scaled)

    plot_PCA_with_clusters(parameters_two_peaks_scaled, SN_labels, kmeans, best_number, number_of_peaks)

    # %%

    # #### model parameters only one peak

    # one_peak = np.where(number_of_peaks == 1)

    # parameters_two_peaks_names = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
    #                             "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"] \
    #                             + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]

    # parameters_two_peaks_scaled = scaler.fit_transform(fitting_parameters[one_peak, 1:][0])

    # plot_PCA(parameters_two_peaks_scaled, SN_labels[one_peak], parameters_two_peaks_names)
    # best_number = number_of_clusters(parameters_two_peaks_scaled)

    # kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    # kmeans.fit(parameters_two_peaks_scaled)

    # plot_PCA_with_clusters(parameters_two_peaks_scaled, SN_labels[one_peak], kmeans, best_number, number_of_peaks[one_peak])

    # %%

    # model parameteres one peak fit

    parameters_one_peak_names = ["A_f1", "t_0_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                                "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"] \

    parameters_two_peaks_scaled = scaler.fit_transform(fitting_parameters_one_peak[:, 1:])

    plot_PCA(parameters_two_peaks_scaled, SN_labels, parameters_one_peak_names)
    best_number = number_of_clusters(parameters_two_peaks_scaled)
    # best_number = 2

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(parameters_two_peaks_scaled)

    plot_PCA_with_clusters(parameters_two_peaks_scaled, SN_labels, kmeans, best_number, [1] * len(number_of_peaks))

    # %%

    # global parameters

    global_parameters_names = ["peak_mag_r", "rise_time_r", "mag_diff_10_r", "mag_diff_15_r", \
                            "mag_diff_30_r", "duration_50_r", "duration_20_r", \
                            "peak_mag_g", "rise_time_g", "mag_diff_10_g", "mag_diff_15_g", \
                            "mag_diff_30_g", "duration_50_g", "duration_20_g"] #, "z"]

    # global_parameters_redshift = np.concatenate((global_parameters, np.array(redshifts).reshape(len(SN_labels[]), 1)), axis = 1)

    global_parameters_scaled = scaler.fit_transform(global_parameters)

    plot_PCA(global_parameters_scaled, SN_labels, global_parameters_names)
    best_number = number_of_clusters(global_parameters_scaled)

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(global_parameters_scaled)

    plot_PCA_with_clusters(global_parameters_scaled, SN_labels, kmeans, best_number, number_of_peaks)

    # %%

    # # combine the two

    # # one_peak = np.where(number_of_peaks == 1)

    # combination_parameters_names = parameters_two_peaks_names + global_parameters_names

    # combination_parameters = np.hstack([fitting_parameters[:, 1:], global_parameters])

    # combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    # plot_PCA(combination_parameters_scaled, SN_labels, combination_parameters_names)
    # best_number = number_of_clusters(combination_parameters_scaled)

    # kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    # kmeans.fit(combination_parameters_scaled)

    # plot_PCA_with_clusters(combination_parameters_scaled, SN_labels, kmeans, best_number, number_of_peaks)

    # %%

    # combine the two one-peak fit

    combination_parameters_names = parameters_one_peak_names  + global_parameters_names

    combination_parameters = np.hstack([fitting_parameters_one_peak[:, 1:], global_parameters_one_peak])

    combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    plot_PCA(combination_parameters_scaled, SN_labels, combination_parameters_names)
    best_number = number_of_clusters(combination_parameters_scaled)

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(combination_parameters_scaled)

    plot_PCA_with_clusters(combination_parameters_scaled, SN_labels, kmeans, best_number, number_of_peaks)

    # %%

    # # combine the two no two-peak

    # one_peak = np.where(number_of_peaks == 1)

    # combination_parameters_names = parameters_two_peaks_names  + global_parameters_names

    # combination_parameters = np.hstack([fitting_parameters[one_peak, 1:][0], global_parameters[one_peak]])

    # combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    # plot_PCA(combination_parameters_scaled, SN_labels[one_peak], combination_parameters_names)
    # best_number = number_of_clusters(combination_parameters_scaled)

    # kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    # kmeans.fit(combination_parameters_scaled)

    # plot_PCA_with_clusters(combination_parameters_scaled, SN_labels[one_peak], kmeans, best_number, number_of_peaks[one_peak])

    # %%
    list(colours.values())
    # %%

    #### low dimension + fitting parameters

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters[:, 1:])

    new_dimensions = loss_of_information(fitting_parameters_scaled, 50) #, f"{survey}_fit_parameters_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_fitting_parameters = dimensionality_reduction(fitting_parameters_scaled, new_dimensions)

    best_number = number_of_clusters(low_dimension_fitting_parameters) #, f"{survey}_fit_parameters_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_fitting_parameters)

    plot_PCA_with_clusters(low_dimension_fitting_parameters, SN_labels, kmeans, best_number, number_of_peaks) #, f"{survey}_fit_parameters_in_the_PC_space")

    # %%

    #### low dimension + fitting parameters one peak fit

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters_one_peak[:, 1:])

    new_dimensions = loss_of_information(fitting_parameters_scaled, 50, f"{survey}_one-peak_fit_parameters_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_fitting_parameters = dimensionality_reduction(fitting_parameters_scaled, new_dimensions)

    best_number = number_of_clusters(low_dimension_fitting_parameters, f"{survey}_one-peak_fit_parameters_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_fitting_parameters)

    print(len(np.where(kmeans.labels_ == 1)[0]))

    plot_PCA_with_clusters(low_dimension_fitting_parameters, SN_labels, kmeans, best_number, [1] * len(number_of_peaks), f"{survey}_one-peak_fit_parameters_in_the_PC_space")

    # %%

    #### low dimension + global parameters

    global_parameters_scaled = scaler.fit_transform(global_parameters_one_peak)

    new_dimensions = loss_of_information(global_parameters_scaled, 50, f"{survey}_light_curve_properties_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_global_parameters = dimensionality_reduction(global_parameters_scaled, new_dimensions)

    best_number = number_of_clusters(low_dimension_global_parameters, f"{survey}_light_curve_properties_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_global_parameters)

    print(len(np.where(kmeans.labels_ == 1)[0]))

    plot_PCA_with_clusters(low_dimension_global_parameters, SN_labels, kmeans, best_number, number_of_peaks, f"{survey}_light_curve_properties_in_the_PC_space")

    # %%

    #### low dimension + combine the two

    combination_parameters_names = parameters_two_peaks_names  + global_parameters_names

    combination_parameters = np.hstack([fitting_parameters[:, 1:], global_parameters])

    combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    new_dimensions = loss_of_information(combination_parameters_scaled, 50, f"{survey}_combined_dataset_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_combination_parameters = dimensionality_reduction(combination_parameters_scaled, new_dimensions)

    best_number = number_of_clusters(low_dimension_combination_parameters, f"{survey}_combined_dataset_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_combination_parameters)

    plot_PCA_with_clusters(low_dimension_combination_parameters, SN_labels, kmeans, best_number, number_of_peaks, f"{survey}_combined_dataset_in_the_PC_space")

    # %%

    #### low dimension + combine the two one-peak fit

    # combination_parameters_names = parameters_one_peak_names  + global_parameters_names

    combination_parameters = np.hstack([fitting_parameters_one_peak[:, 1:], global_parameters_one_peak])

    combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    new_dimensions = loss_of_information(combination_parameters_scaled, 50) # , f"{survey}_combined_dataset_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_combination_parameters = dimensionality_reduction(combination_parameters_scaled, new_dimensions)

    best_number = number_of_clusters(low_dimension_combination_parameters) # , f"{survey}_combined_dataset_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_combination_parameters)

    print(len(np.where(kmeans.labels_ == 2)[0]))

    plot_PCA_with_clusters(low_dimension_combination_parameters, SN_labels, kmeans, best_number, number_of_peaks) # , f"{survey}_combined_dataset_in_the_PC_space")

    # %%

    # #### low dimension + combine the two one-peak

    # # combination_parameters_names = parameters_two_peaks_names  + global_parameters_names

    # combination_parameters = np.hstack([fitting_parameters[one_peak, 1:][0], global_parameters[one_peak]])

    # combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    # new_dimensions = loss_of_information(combination_parameters_scaled, 50)
    # print("best number of dimensions", new_dimensions)
    # low_dimension_combination_parameters = dimensionality_reduction(combination_parameters_scaled, new_dimensions)

    # best_number = number_of_clusters(low_dimension_combination_parameters)

    # kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    # kmeans.fit(low_dimension_combination_parameters)

    # plot_PCA_with_clusters(low_dimension_combination_parameters, SN_labels[one_peak], kmeans, best_number, number_of_peaks[one_peak])

    # %%

    # collection_times_f1, collection_times_f2, collection_fluxes_f1, collection_fluxes_f2 = plot_SN_collection(survey, fitting_parameters[one_peak], number_of_peaks[one_peak])
    collection_times_f1, collection_times_f2, collection_fluxes_f1, collection_fluxes_f2 = plot_SN_collection(survey, fitting_parameters_one_peak, [1] * len(number_of_peaks))

    cluster_0 = np.where(kmeans.labels_ == 0)
    cluster_1 = np.where(kmeans.labels_ == 1)
    cluster_2 = np.where(kmeans.labels_ == 2)
    # cluster_3 = np.where(kmeans.labels_ == 3)

    # plt.plot(np.mean(np.array(collection_times_f1)[cluster_3], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_3], axis = 0), linewidth = 2, color = "#76B7B2", label = "K-means cluster 3")
    # plt.fill_between(np.mean(np.array(collection_times_f1)[cluster_3], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_3], axis = 0) - np.std(np.array(collection_fluxes_f1)[cluster_3], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_3], axis = 0) + np.std(np.array(collection_fluxes_f1)[cluster_3], axis = 0), color = "tab:cyan", alpha = 0.15)
    
    # plt.plot(np.mean(np.array(collection_times_f1)[cluster_2], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_2], axis = 0), linewidth = 2, color = "#9C755F", label = "K-means cluster 3")
    # plt.fill_between(np.mean(np.array(collection_times_f1)[cluster_2], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_2], axis = 0) - np.std(np.array(collection_fluxes_f1)[cluster_2], axis = 0), np.mean(np.array(collection_fluxes_f1)[cluster_2], axis = 0) + np.std(np.array(collection_fluxes_f1)[cluster_2], axis = 0), color = "tab:brown", alpha = 0.15)

    # plt.plot(np.mean(np.array(collection_times_f2)[cluster_1], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_1], axis = 0), linewidth = 2, color = "#B07AA1", label = "K-means cluster 2")
    # plt.fill_between(np.mean(np.array(collection_times_f2)[cluster_1], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_1], axis = 0) - np.std(np.array(collection_fluxes_f2)[cluster_1], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_1], axis = 0) + np.std(np.array(collection_fluxes_f2)[cluster_1], axis = 0), color = "tab:purple", alpha = 0.15)

    # plt.plot(np.mean(np.array(collection_times_f2)[cluster_0], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_0], axis = 0), linewidth = 2, color = "#59A14F", label = "K-means cluster 1")
    # plt.fill_between(np.mean(np.array(collection_times_f2)[cluster_0], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_0], axis = 0) - np.std(np.array(collection_fluxes_f2)[cluster_0], axis = 0), np.mean(np.array(collection_fluxes_f2)[cluster_0], axis = 0) + np.std(np.array(collection_fluxes_f2)[cluster_0], axis = 0), color = "tab:green", alpha = 0.15)

    # plt.scatter(np.array(collection_times_f1)[cluster_3], np.array(collection_fluxes_f1)[cluster_3], s = 1, color = colours["cyan"], label = "K-means cluster 3")
    # plt.legend()
    # plt.xlim([-200, 500])
    # plt.show()

    # plt.scatter(np.array(collection_times_f1)[cluster_2], np.array(collection_fluxes_f1)[cluster_2], s = 1, color = colours["brown"], label = "K-means cluster 2")
    # plt.legend()
    # plt.xlim([-200, 500])
    # plt.show()

    # plt.scatter(np.array(collection_times_f1)[cluster_1], np.array(collection_fluxes_f1)[cluster_1], s = 1, color = colours["purple"], label = "K-means cluster 1")
    # plt.legend()
    # plt.xlim([-200, 500])
    # plt.show()

    plt.scatter(np.array(collection_times_f1)[cluster_0], np.array(collection_fluxes_f1)[cluster_0], s = 1, color = colours["green"], label = "K-means cluster 0")
    # plt.legend()
    # plt.xlim([-200, 500])
    # plt.show()

    plt.xlabel("Time since peak (days)", fontsize = 13)
    plt.ylabel("Normalized flux", fontsize = 13)
    plt.title(f"Normalized {survey} r-band light curves.")
    plt.grid(alpha = 0.3) 
    plt.legend()
    # plt.savefig(f"Plots/Results/light_curve_template_{survey}_combined_dataset_in_the_PC_space", dpi = 300, bbox_inches = "tight")
    plt.show()

    # %%

    for name in fitting_parameters[cluster_1, 0][0]:
        print(name)
        time, flux, _, filters = ztf_load_data(name)
        f1_values = np.where(filters == "r")

        # Shift the light curve so that the main peak is at time = 0 MJD
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

        time -= peak_time
        flux /= peak_flux

        plt.scatter(time[f1_values], flux[f1_values], s = 1, color = "#B07AA1" )

    plt.show()

    # %%

    cluster_0 = np.where(kmeans.labels_ == 0)
    cluster_1 = np.where(kmeans.labels_ == 1)
    cluster_2 = np.where(kmeans.labels_ == 2)

    # print(fitting_parameters[one_peak, 0][0][cluster_0])
    # print(fitting_parameters[one_peak, 0][0][cluster_1])

    print(fitting_parameters_one_peak[:, 0][cluster_0], len(fitting_parameters_one_peak[:, 0][cluster_0]))
    print(fitting_parameters_one_peak[:, 0][cluster_1], len(fitting_parameters_one_peak[:, 0][cluster_1]))
    print(fitting_parameters_one_peak[:, 0][cluster_2], len(fitting_parameters_one_peak[:, 0][cluster_2]))

    # print(fitting_parameters[:, 0][cluster_0])
    # print(fitting_parameters[:, 0][cluster_1])
        
    print(SN_labels[cluster_0])
    print(SN_labels[cluster_1])
    print(SN_labels[cluster_2])

    # %%

    print(np.std(global_parameters_one_peak[cluster_1, 4][0]/30))
    
    # %%

    for idx in cluster_1[0]:
        
        f1 = "r"
        f2 = "g"

        time, flux, _, filters = load_ztf_data(fitting_parameters[idx, 0])

        f1_values = np.where(filters == f1)
        f2_values = np.where(filters == f2)

        # Shift the light curve so that the main peak is at time = 0 MJD
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

        time -= peak_time

        # Reshape the data so that the flux is between 0 and 1 micro Jy
        # flux_fit_min_f1 = np.copy(np.min(flux_fit[f1_values_fit]))
        flux_max_f1 = np.copy(np.max(flux[f1_values]))
        flux_f1 = (flux[f1_values]) / flux_max_f1

        # # flux_min_f2 = np.copy(np.min(flux[f2_values]))
        # flux_max_f2 = np.copy(np.max(flux[f2_values]))
        # flux_f2 = (flux[f2_values]) / flux_max_f2

        plt.scatter(time[f1_values], flux_f1, s = 1, c = "#4E79A7")

    for idx in cluster_0[0]:
        
        f1 = "r"
        f2 = "g"

        time, flux, _, filters = load_ztf_data(fitting_parameters[idx, 0])

        f1_values = np.where(filters == f1)
        f2_values = np.where(filters == f2)

        # Shift the light curve so that the main peak is at time = 0 MJD
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

        time -= peak_time

        # Reshape the data so that the flux is between 0 and 1 micro Jy
        # flux_fit_min_f1 = np.copy(np.min(flux_fit[f1_values_fit]))
        flux_max_f1 = np.copy(np.max(flux[f1_values]))
        flux_f1 = (flux[f1_values]) / flux_max_f1

        # # flux_min_f2 = np.copy(np.min(flux[f2_values]))
        # flux_max_f2 = np.copy(np.max(flux[f2_values]))
        # flux_f2 = (flux[f2_values]) / flux_max_f2

        plt.scatter(time[f1_values], flux_f1, s = 1, c = "tab:orange")



    # %%
    # ["ZTF19aceqlxc", "ZTF19acykaae", "ZTF18aamftst"] in S23 as possible Ia-CSM and also in cluster 0
    # "ZTF19acvkibv" in S23 as possible Ia-CSM but not in cluster 0 0

    # ["2019kep"] in S23 as possible Ia-CSM 
    # Add well known SNe

    # mag_diff_15_r mag_diff_30_r duration_50_r duration_20_r mag_diff_15_g mag_diff_30_g duration_20_g
    # The shape of the light curve afterwards is important
    # Maybe it removes plateau LC


    # Below t-SNE and UMAP
    # %%

    # # tSNE

    # def plot_tSNE(parameter_values, SN_type):

    #     model = TSNE(n_components = 2, random_state = 2804)

    #     tsne_data = model.fit_transform(parameter_values)
    #     tsne_df = pd.DataFrame(data = tsne_data, columns = ("Dimension 1", "Dimension 2"))
    #     tsne_df["SN_type"] = SN_type

    #     plt.figure(figsize=(8, 6))

    #     colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    #     unique_labels = np.unique(SN_type)
    #     for i, lbl in enumerate(unique_labels):

    #         class_data = tsne_df[tsne_df["SN_type"] == lbl]
    #         plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

    #     plt.title(f"t-SNE plot")
    #     plt.xlabel("Dimension 1")
    #     plt.ylabel("Dimension 2")
    #     plt.grid(alpha = 0.3)
    #     plt.legend()
    #     plt.show()

    # # %%

    # # UMAP

    # def plot_UMAP(parameter_values, SN_type):

    #     reducer = umap.UMAP(random_state = 2804)

    #     UMAP_data = reducer.fit_transform(parameter_values)

    #     UMAP_df = pd.DataFrame(data = UMAP_data, columns = ("Dimension 1", "Dimension 2"))
    #     UMAP_df["SN_type"] = SN_type

    #     plt.figure(figsize=(8, 6))

    #     colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    #     unique_labels = np.unique(SN_type)
    #     for i, lbl in enumerate(unique_labels):

    #         class_data = UMAP_df[UMAP_df["SN_type"] == lbl]
    #         plt.scatter(class_data["Dimension 1"], class_data["Dimension 2"], c = colors[i], label = lbl)

    #     plt.title(f"UMAP plot")
    #     plt.xlabel("Dimension 1")
    #     plt.ylabel("Dimension 2")
    #     plt.grid(alpha = 0.3)
    #     plt.legend()
    #     plt.show()
        



# %%

fitting_parameters_one_peak[cluster_1, 1:][0][0]

# %%
global_parameters_one_peak[166]
# %%

def calculate_global_parameters(SN_id, survey, parameter_values):

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

    flux_fit = light_curve_one_peak(time_fit, parameter_values[:14], peak_flux, f1_values_fit, f2_values_fit)

    magnitude_fit, _ = atlas_micro_flux_to_magnitude(flux_fit, flux_fit)

    time_fit += peak_time

    # In filter 1
    peak_fit_idx_f1 = np.argmax(flux_fit[f1_values_fit])
    peak_time_fit_f1 = np.copy(time_fit[peak_fit_idx_f1])
    peak_flux_fit_f1 = np.copy(flux_fit[peak_fit_idx_f1])
    time_fit_f1 = time_fit - peak_time_fit_f1

    # Peak magnitude
    peak_magnitude_f1 = magnitude_fit[peak_fit_idx_f1]

    # Magnitude difference peak - 15 days after
    fifteen_days_after_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] - 50))
    fifteen_days_magnitude_f1 = magnitude_fit[fifteen_days_after_peak_f1]
    fifteen_days_magnitude_difference_f1 = np.abs(peak_magnitude_f1 - fifteen_days_magnitude_f1)

    # Magnitude difference peak - 30 days after
    thirty_days_after_peak_f1 = np.argmin(np.abs(time_fit_f1[f1_values_fit] - 100))
    thirty_days_magnitude_f1 = magnitude_fit[thirty_days_after_peak_f1]
    thirty_days_magnitude_difference_f1 = np.abs(peak_magnitude_f1 - thirty_days_magnitude_f1)

    # In filter 2
    peak_fit_idx_f2 = np.argmax(flux_fit[f2_values_fit]) + amount_fit
    peak_time_fit_f2 = np.copy(time_fit[peak_fit_idx_f2])
    peak_flux_fit_f2 = np.copy(flux_fit[peak_fit_idx_f2])
    time_fit_f2 = time_fit - peak_time_fit_f2

    # Peak magnitude
    peak_magnitude_f2 = magnitude_fit[peak_fit_idx_f2]

    # Magnitude difference peak - 15 days after
    fifteen_days_after_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] - 50)) + amount_fit
    fifteen_days_magnitude_f2 = magnitude_fit[fifteen_days_after_peak_f2]
    fifteen_days_magnitude_difference_f2 = np.abs(peak_magnitude_f2 - fifteen_days_magnitude_f2)

    # Magnitude difference peak - 30 days after
    thirty_days_after_peak_f2 = np.argmin(np.abs(time_fit_f2[f2_values_fit] - 100)) + amount_fit
    thirty_days_magnitude_f2 = magnitude_fit[thirty_days_after_peak_f2]
    thirty_days_magnitude_difference_f2 = np.abs(peak_magnitude_f2 - thirty_days_magnitude_f2)

    return np.array([fifteen_days_magnitude_difference_f1, thirty_days_magnitude_difference_f1, \
                     fifteen_days_magnitude_difference_f2, thirty_days_magnitude_difference_f2])

# %%

magnitude_rates = np.array([calculate_global_parameters(fitting_parameters_one_peak[cluster_1, 0][0][idx], survey, fitting_parameters_one_peak[cluster_1, 1:][0][idx]) for idx in range(len(fitting_parameters_one_peak[cluster_1, 0][0]))])
# %%
np.std(magnitude_rates[:, 0]/50)
# %%
