# %%

from data_processing import load_ztf_data, load_atlas_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks
from parameter_reduction import determine_parameters, retrieve_redshift, calculate_peak_absolute_magnitude

from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, metrics
# from sklearn.manifold import TSNE
# import umap.umap_ as umap
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True

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

def plot_PCA_with_clusters(parameter_values, SN_type, kmeans, number_of_peaks):

    pca = decomposition.PCA(n_components = 2, random_state = 2804)
    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data, columns = ["Dimension 1", "Dimension 2"])
    pca_df["SN_type"] = SN_type
    pca_df["number_of_peaks"] = number_of_peaks

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
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, grid_clusters, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], colors=["tab:green", "tab:purple", "tab:brown", "tab:cyan"], alpha=0.3)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    unique_labels = np.unique(SN_type)

    for i, lbl in enumerate(unique_labels):
        # Filter the data for the current label
        class_data = pca_df[pca_df["SN_type"] == lbl]

        # Plot circles for number_of_peaks == 1 and triangles for number_of_peaks == 2
        one_peak_data = class_data[class_data["number_of_peaks"] == 1]
        two_peak_data = class_data[class_data["number_of_peaks"] == 2]

        # Plot one-peak supernovae (circles)
        plt.scatter(one_peak_data["Dimension 1"], one_peak_data["Dimension 2"], 
                    c = colors[i], marker = 'o', label = f'{lbl} (1 peak)', edgecolor = 'k')

        # Plot two-peak supernovae (triangles)
        plt.scatter(two_peak_data["Dimension 1"], two_peak_data["Dimension 2"], 
                    c = colors[i], marker = '^', label = f'{lbl} (2 peaks)', edgecolor = 'k')

    # Create proxy artists for the background clusters
    cluster0_patch = mpatches.Patch(color = "tab:green", label = "K-means cluster 0")
    cluster1_patch = mpatches.Patch(color = "tab:purple", label = "K-means cluster 1")
    cluster2_patch = mpatches.Patch(color = "tab:brown", label = "K-means cluster 2")
    cluster3_patch = mpatches.Patch(color = "tab:cyan", label = "K-means cluster 3")

    # Labels and legend
    plt.title(f"PCA plot")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.3)
    plt.legend(handles = plt.legend().legendHandles + [cluster0_patch, cluster1_patch, cluster2_patch, cluster3_patch])
    plt.show()

    
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

if __name__ == '__main__':
    # %%

    survey = "ZTF"
    fitting_parameters, global_parameters, number_of_peaks, SN_labels, SN_labels_color = determine_parameters(survey)

    scaler = StandardScaler()

    # %%

    #### No redshift

    fitting_parameters_names = ["t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                                "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"]

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters[:, 3:15])

    plot_PCA(fitting_parameters_scaled, SN_labels, fitting_parameters_names)
    best_number = number_of_clusters(fitting_parameters_scaled)

    kmeans = KMeans(n_clusters = best_number)
    kmeans.fit(fitting_parameters_scaled)

    plot_PCA_with_clusters(fitting_parameters_scaled, SN_labels, kmeans, number_of_peaks)

    # %%

    #### With redshift

    redshifts = retrieve_redshift(fitting_parameters[:, 0], survey)

    peak_abs_magnitude = []
    for idx in range(len(redshifts)):

        peak_abs_magnitude.append(calculate_peak_absolute_magnitude(global_parameters[idx, 0], redshifts[idx]))

    fitting_parameters_names = ["A_f1", "t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                                "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2", \
                                "M_r", "z"]

    fitting_parameters_redshift = np.concatenate((fitting_parameters[:, 1].reshape(len(SN_labels), 1), fitting_parameters[:, 3:15], np.array(peak_abs_magnitude).reshape(len(SN_labels), 1), np.array(redshifts).reshape(len(SN_labels), 1)), axis = 1)

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters_redshift)

    plot_PCA(fitting_parameters_scaled, SN_labels, fitting_parameters_names)
    best_number = number_of_clusters(fitting_parameters_scaled)

    kmeans = KMeans(n_clusters = 2)
    kmeans.fit(fitting_parameters_scaled)

    plot_PCA_with_clusters(fitting_parameters_scaled, SN_labels, kmeans, number_of_peaks)

    # %%

    #### No redshift + double peak

    parameters_two_peaks_names = ["t_rise_f1", "gamma_f1", "beta_f1", "t_fall_f1", "error_f1", \
                                "A_f2", "t_0_f2", "t_rise_f2", "gamma_f2", "beta_f2", "t_fall_f2", "error_f2"] \
                                + ["amp_f1", "mu_f1", "std_f1", "amp_f2", "mu_f2", "std_f2"]

    parameters_two_peaks_scaled = scaler.fit_transform(fitting_parameters[:, 3:])

    plot_PCA(parameters_two_peaks_scaled, SN_labels, parameters_two_peaks_names)
    best_number = number_of_clusters(parameters_two_peaks_scaled)

    kmeans = KMeans(n_clusters = best_number)
    kmeans.fit(parameters_two_peaks_scaled)

    plot_PCA_with_clusters(parameters_two_peaks_scaled, SN_labels, kmeans, number_of_peaks)

    # %%

    global_parameters_names = ["peak_mag_r", "rise_time_r", "mag_diff_10_r", "mag_diff_15_r", \
                            "mag_diff_30_r", "duration_50_r", "duration_20_r", \
                            "peak_mag_g", "rise_time_g", "mag_diff_10_g", "mag_diff_15_g", \
                            "mag_diff_30_g", "duration_50_g", "duration_20_g"] #, "z"]

    # global_parameters_redshift = np.concatenate((global_parameters, np.array(redshifts).reshape(len(SN_labels[]), 1)), axis = 1)

    global_parameters_scaled = scaler.fit_transform(global_parameters)

    plot_PCA(global_parameters_scaled, SN_labels, global_parameters_names)
    best_number = number_of_clusters(global_parameters_scaled)

    kmeans = KMeans(n_clusters = best_number)
    kmeans.fit(global_parameters_scaled)

    plot_PCA_with_clusters(global_parameters_scaled, SN_labels, kmeans, number_of_peaks)

    cluster_0 = np.where(kmeans.labels_ == 0)
    cluster_1 = np.where(kmeans.labels_ == 1)
    cluster_2 = np.where(kmeans.labels_ == 2)
    cluster_3 = np.where(kmeans.labels_ == 3)

# %%

    global_parameters_scaled_cluster_0 = scaler.fit_transform(global_parameters[cluster_0])

    plot_PCA(global_parameters_scaled_cluster_0, SN_labels[cluster_0], global_parameters_names)
    best_number = number_of_clusters(global_parameters_scaled_cluster_0)

    kmeans = KMeans(n_clusters = best_number)
    kmeans.fit(global_parameters_scaled_cluster_0)

    plot_PCA_with_clusters(global_parameters_scaled_cluster_0, SN_labels[cluster_0], kmeans, number_of_peaks[cluster_0])

    # %%

    # global_parameters_names_reduced = ["mag_diff_10_r", "mag_diff_15_r", \
    #                         "mag_diff_30_r", "duration_50_r", "duration_20_r", \
    #                         "mag_diff_10_g", "mag_diff_15_g", \
    #                         "mag_diff_30_g", "duration_50_g", "duration_20_g"] #, "z"]

    # # global_parameters_redshift = np.concatenate((global_parameters, np.array(redshifts).reshape(len(SN_labels[]), 1)), axis = 1)

    # global_parameters_scaled_reduced = np.concatenate((global_parameters_scaled[:, 2:7], global_parameters_scaled[:, 9:]), axis = 1)

    # plot_PCA(global_parameters_scaled_reduced, SN_labels, global_parameters_names_reduced)
    # best_number = number_of_clusters(global_parameters_scaled_reduced)

    # kmeans = KMeans(n_clusters = best_number)
    # kmeans.fit(global_parameters_scaled_reduced)

    # plot_PCA_with_clusters(global_parameters_scaled_reduced, SN_labels, kmeans, number_of_peaks)

    # %%

    # combine the two

    # %%

    collection_times_f1, collection_times_f2, collection_fluxes_f1, collection_fluxes_f2 = plot_SN_collection(fitting_parameters, number_of_peaks)

    # %%

    cluster_0 = np.where(kmeans.labels_ == 0)
    cluster_1 = np.where(kmeans.labels_ == 1)
    # plt.scatter(np.array(collection_times_f1)[cluster_3], np.array(collection_fluxes_f1)[cluster_3], s = 1, label = "cluster 3")
    # plt.legend()
    # plt.show()
    # plt.scatter(np.array(collection_times_f1)[cluster_2], np.array(collection_fluxes_f1)[cluster_2], s = 1, label = "cluster 2")
    # plt.legend()
    # plt.show()
    plt.scatter(np.array(collection_times_f1)[cluster_1], np.array(collection_fluxes_f1)[cluster_1], s = 1, label = "cluster 1")
    # plt.legend()
    # plt.show()
    plt.scatter(np.array(collection_times_f1)[cluster_0], np.array(collection_fluxes_f1)[cluster_0], s = 1, label = "cluster 0")
    plt.legend()
    plt.show()

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

        plt.scatter(time[f1_values], flux_f1, s = 1, c = "tab:blue")

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
    plt.hist(redshifts)
    # %%

# %%
