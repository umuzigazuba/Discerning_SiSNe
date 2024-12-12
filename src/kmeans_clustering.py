# %%

__ImportError__ = "One or more required packages are not installed. See requirements.txt."

try:
    from data_processing import ztf_load_data, atlas_load_data
    from parameter_estimation import light_curve_one_peak, light_curve_two_peaks

    from sklearn.preprocessing import StandardScaler
    from sklearn import decomposition, metrics
    from sklearn.cluster import KMeans
    from sklearn.model_selection import ParameterGrid

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd

except ImportError:
    raise ImportError(__ImportError__)

plt.rcParams["text.usetex"] = True
plt.rcParams['axes.axisbelow'] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

# PCA

def important_feature_selection(parameter_values, sn_type, parameter_names):

    """
    Print the most informational parameters

    Parameters: 
       parameter_values (numpy.ndarray): Parameters describing the SNe
       sn_type (list): List of SN type
       parameter_names (list): List of parameter names

    Outputs: 
        None
    """

    # Reduce the dimensionality of the parameter space
    pca = decomposition.PCA(n_components = 2, random_state = 2804)
    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data, columns = ("Dimension 1", "Dimension 2"))   
    pca_df["sn_type"] = sn_type
        
    print("Explained variation per principal component: {}".format(pca.explained_variance_ratio_))
    print("Cumulative variance explained by 2 principal components: {:.2%}".format(np.sum(pca.explained_variance_ratio_)))

    dataset_pca = pd.DataFrame(abs(pca.components_), columns = parameter_names, index = ["PC_1", "PC_2"])
    print("\n\n", dataset_pca)

    print("Most important features PC 1:\n", (dataset_pca[dataset_pca > 0.25].iloc[0]).dropna())
    print("\n\nMost important features PC 2:\n", (dataset_pca[dataset_pca > 0.25].iloc[1]).dropna())

def plot_PCA_with_clusters(parameter_values, sn_type, kmeans, best_number, number_of_peaks, save_fig = False):

    """
    Plot the SN parameter space and the clusters found by the KMeans aglorithm in two-dimensions using PCA

    Parameters: 
       parameter_values (numpy.ndarray): Parameters describing the SNe
       sn_type (list): List of SN type
       kmeans (object): Kmeans model
       best_number (int): Best number of clusters the parameter space can be divided into
       number_of_peaks (numpy.ndarray): Number of peaks of the used model 
       save_fig (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs: 
        None
    """

    pca = decomposition.PCA(n_components = 2, random_state = 2804)
    pca_data = pca.fit_transform(parameter_values)

    pca_df = pd.DataFrame(data = pca_data[:, :2], columns = ["Dimension 1", "Dimension 2"])
    pca_df["sn_type"] = sn_type
    pca_df["number_of_peaks"] = number_of_peaks

    # Create a mesh grid to cover the PCA space
    x_min, x_max = pca_data[:, 0].min() - 1, pca_data[:, 0].max() + 1
    y_min, y_max = pca_data[:, 1].min() - 1, pca_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))

    # Predict the clusters for each point in the mesh grid
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Inverse transform the PCA grid points back to the original parameter space
    grid_original = pca.inverse_transform(grid_2d)

    # Predict the clusters using KMeans on the original parameter space
    grid_clusters = kmeans.predict(grid_original)
    grid_clusters = grid_clusters.reshape(xx.shape)

    cluster_colours = {"green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE"}
    clusters = [mpatches.Patch(color = list(cluster_colours.values())[idx], label = f"K-means cluster {idx + 1}") for idx in range(best_number)]

    # Plot the cluster regions 
    plt.figure(figsize = (8, 6))
    plt.contourf(xx, yy, grid_clusters, levels = [-0.5, 0.5, 1.5, 2.5, 3.5], colors = list(cluster_colours.values()), alpha = 0.3)

    unique_labels = np.unique(sn_type)
    for i, lbl in enumerate(unique_labels):

        # Filter the data for the current label
        class_data = pca_df[pca_df["sn_type"] == lbl]

        # Plot circles for one-peak light curves and triangles for two-peak light curves
        one_peak_data = class_data[class_data["number_of_peaks"] == 1]
        two_peak_data = class_data[class_data["number_of_peaks"] == 2]

        if len(one_peak_data) != 0:
            plt.scatter(one_peak_data["Dimension 1"], one_peak_data["Dimension 2"], s = 65,
                        c = list(colours.values())[i], marker = 'o', label = f'{lbl} (one peak)', edgecolor = 'k')

        if len(two_peak_data) != 0:
            plt.scatter(two_peak_data["Dimension 1"], two_peak_data["Dimension 2"], s = 65,
                        c = list(colours.values())[i], marker = '^', label = f'{lbl} (two peaks)', edgecolor = 'k')

    # Labels and legend
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha = 0.5)
    plt.legend()
    plt.legend(handles = plt.legend().legendHandles + clusters)

    if type(save_fig) == str:
        title = save_fig.replace("_", " ")
        plt.title(f"K-means clusters of the {title}.")
        plt.savefig(f"../plots/machine_learning/PCA_plot_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"PCA plot")
        plt.show()
  
def dimensionality_reduction(parameter_values, new_dimensions):

    """
    Reduce the dimensionality of a dataset

    Parameters: 
       parameter_values (numpy.ndarray): Parameters describing the SNe
       new_dimensions (int): New number of dimensions of the parameter space

    Outputs:
        low_dimension_parameter_values: Parameter set reduced in dimensionality
    """

    pca = decomposition.PCA(n_components = new_dimensions, random_state = 2804)

    low_dimension_parameter_values = pca.fit_transform(parameter_values)

    return low_dimension_parameter_values

# %%

def silhouette_score(parameter_values, save_fig = False):

    """
    Determine the best number of clusters a dataset can be divided into using the silhouette score

    Parameters: 
        parameter_values (numpy.ndarray): Parameters describing the SNe
        save_fig (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs:
        best_number (int): Best number of clusters
    """

    n_clusters = np.array([2, 3, 4])
    silhouette_scores = []

    parameter_grid = ParameterGrid({"n_clusters": n_clusters})
    best_score = -1

    kmeans_model = KMeans(random_state = 2804)

    for p in parameter_grid:
        
        kmeans_model.set_params(**p)    
        kmeans_model.fit(parameter_values)

        ss = metrics.silhouette_score(parameter_values, kmeans_model.labels_)
        silhouette_scores += [ss] 
        if ss > best_score:
            best_score = ss

    mask = np.where(silhouette_scores == best_score)
    best_number = n_clusters[mask][0]

    plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align = "center", color = colours["blue"], width = 0.5)
    plt.xticks(range(len(silhouette_scores)), list(n_clusters))
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.grid(alpha = 0.5)

    if type(save_fig) == str:
        title = save_fig.replace("_", " ")
        plt.title(f"Silhouette score of the {title}.")
        plt.savefig(f"../plots/machine_learning/Silhouette_score_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"Silhouette score. Best number of clusters = {best_number}. Score = {round(best_score, 2)}.")
        plt.show()

    return best_number

def loss_of_information(parameter_values, percentage, save_fig = False):

    """
    Determine the dimesionality that retains a certain percentage of the orginal dimensionaity's information

    Parameters: 
        parameter_values (numpy.ndarray): Parameters describing the SNe
        percentage (float): The percentage of information that needs to remain
        save_fig (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs:
        best_dimensionality (int): Best dimensionality
    """

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
    plt.grid(alpha = 0.5)

    if type(save_fig) == str:
        name = save_fig.replace("_", " ")
        plt.title(f"Cumulative variance of the {name}.")
        plt.savefig(f"../plots/machine_learning/cummulative_variance_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"Cumulative variance.")
        plt.show()

    best_dimensionality_idx = np.argmin(np.abs(np.array(cummulative_variance) - percentage / 100))
    best_dimensionality = possible_dimensions[best_dimensionality_idx]

    return best_dimensionality

# %%

def load_best_fit_light_curves(survey, parameter_values, number_of_peaks):

    """
    Load the best-fit light curves for a set of SNe and their best-fit parameters

    Parameters:
        survey (str): Survey that observed the SNe 
        parameter_values (numpy.ndarray): The best-fit parameter values 
        number_of_peaks (list): The number of peaks of the model that best fit the SN light curves

    Outputs:
        sample_times_f1 (numpy.ndarray): Modified Julian Date of the best-fit data points in the f1 filter
        sample_fluxes_f1 (numpy.ndarray): Differential flux of the best-fit data points in the f1 fitler
        sample_times_f2 (numpy.ndarray): Modified Julian Date of the best-fit data points in the f2 filter
        sample_fluxes_f2 (numpy.ndarray): Differential flux of the best-fit data points in the f2 fitler
    """

    sample_times_f1 = []
    sample_fluxes_f1 = []

    sample_times_f2 = []
    sample_fluxes_f2 = []

    for idx in range(len(parameter_values)):
    
        if survey == "ZTF":
            f1 = "r"

            # Load the data
            time, flux, _, filters = ztf_load_data(parameter_values[idx, 0])

        if survey == "ATLAS":
            f1 = "o"

            # Load the data
            time, flux, _, filters = atlas_load_data(parameter_values[idx, 0])

        f1_values = np.where(filters == f1)

        # Shift the light curve so that the main peak is at 0 MJD
        peak_main_idx = np.argmax(flux[f1_values])
        peak_time = np.copy(time[peak_main_idx])
        peak_flux = np.copy(flux[peak_main_idx])

        time -= peak_time

        amount_fit = 500
        time_fit = np.concatenate((np.linspace(-150, 400, amount_fit), np.linspace(-150, 400, amount_fit)))
        f1_values_fit = np.arange(amount_fit)
        f2_values_fit = np.arange(amount_fit) + amount_fit

        if number_of_peaks[idx] == 1:
            flux_fit = light_curve_one_peak(time_fit, parameter_values[idx, 1:15], peak_flux, f1_values_fit, f2_values_fit)

        elif number_of_peaks[idx] == 2:
            flux_fit = light_curve_two_peaks(time_fit, parameter_values[idx, 1:], peak_flux, f1_values_fit, f2_values_fit)

        time_fit += peak_time

        # Shift the best-fit light curve so that its main peak is at 0 MJD
        peak_main_idx_f1 = np.argmax(flux_fit[f1_values_fit])
        time_fit_f1 = time_fit[f1_values_fit] - time_fit[peak_main_idx_f1]
        sample_times_f1.append(np.array(time_fit_f1))

        peak_main_idx_f2 = np.argmax(flux_fit[f1_values_fit])
        time_fit_f2 = time_fit[f2_values_fit] - time_fit[peak_main_idx_f2]
        sample_times_f2.append(np.array(time_fit_f2))

        # Reshape the data so that the flux is between 0 and 1 micro Jy
        flux_fit_f1 = flux_fit[f1_values_fit] / flux_fit[peak_main_idx_f1]
        sample_fluxes_f1.append(np.array(flux_fit_f1))

        flux_fit_f2 = flux_fit[f2_values_fit] / flux_fit[peak_main_idx_f2]
        sample_fluxes_f2.append(np.array(flux_fit_f2))

    return sample_times_f1, sample_fluxes_f1, sample_times_f2, sample_fluxes_f2

def plot_light_curve_template(kmeans, best_number, sample_times, sample_fluxes, filter, save_fig =  False):

    """
    Plot the light curve template in a certain filter

    Parameters:
        kmeans (object): Kmeans model
        best_number (int): Best number of clusters the parameter space can be divided into
        sample_times (numpy.ndarray): Modified Julian Date of the best-fit data points in a certain filter
        sample_fluxes (numpy.ndarray): Differential flux of the best-fit data points in a certain fitler
        filter (str): Filter used to make the observations
        save_fig (boolean or str): If str, name of the directory where the plot is saved, if boolean, show the plot

    Outputs:
        None
    """

    cluster_colours = {"green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE"}
    for idx in range(best_number): 

        cluster_idx = np.where(kmeans.labels_ == idx)

        mean_time = np.mean(np.array(sample_times)[cluster_idx], axis = 0)

        mean_flux = np.mean(np.array(sample_fluxes)[cluster_idx], axis = 0)
        std_flux = np.std(np.array(sample_fluxes)[cluster_idx], axis = 0)

        plt.plot(mean_time, mean_flux, linewidth = 2, color = list(cluster_colours.values())[idx], label = f"K-means cluster {idx + 1}")
        plt.fill_between(mean_time, mean_flux - std_flux, mean_flux + std_flux, color = list(cluster_colours.values())[idx], alpha = 0.15)
 
    plt.xlabel("Time since peak (days)", fontsize = 13)
    plt.ylabel("Normalised flux", fontsize = 13)
    plt.grid(alpha = 0.3) 
    plt.legend()

    if type(save_fig) == str:
        title = save_fig.replace("_", " ")
        plt.title(f"Normalised light curves in the {filter}-band {title}.")
        plt.savefig(f"../plots/machine_learning/Light_curve_template_{save_fig}", dpi = 300, bbox_inches = "tight")
        plt.show()

    else:
        plt.title(f"Normalised {survey} light curves in the {filter}-band.")
        plt.show()
    
# %%

if __name__ == '__main__':

    survey = "ZTF"
    f1 = "r"
    f2 = "g"

    # survey = "ATLAS"
    # f1 = "o"
    # f2 = "c"
    
    fitting_parameters = np.load(f"../data/machine_learning/{survey}/fitting_parameters.npy", allow_pickle = True)
    fitting_parameters_one_peak = np.load(f"../data/machine_learning/{survey}/fitting_parameters_one_peak.npy", allow_pickle = True)
    global_parameters = np.load(f"../data/machine_learning/{survey}/global_parameters.npy")
    global_parameters_one_peak = np.load(f"../data/machine_learning/{survey}/global_parameters_one_peak.npy")
    number_of_peaks = np.load(f"../data/machine_learning/{survey}/number_of_peaks.npy")
    sn_labels = np.load(f"../data/machine_learning/{survey}/SN_labels.npy")
    sn_labels_color = np.load(f"../data/machine_learning/{survey}/SN_labels_color.npy")
    
    scaler = StandardScaler()

    # %%

    #### low dimension + fitting parameters

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters[:, 1:])

    new_dimensions = loss_of_information(fitting_parameters_scaled, 50, f"{survey}_best-fit_parameters_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_fitting_parameters = dimensionality_reduction(fitting_parameters_scaled, new_dimensions)

    best_number = silhouette_score(low_dimension_fitting_parameters, f"{survey}_best-fit_parameters_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_fitting_parameters)

    plot_PCA_with_clusters(low_dimension_fitting_parameters, sn_labels, kmeans, best_number, number_of_peaks, f"{survey}_best-fit_parameters_in_the_PC_space")

    # %%

    #### low dimension + fitting parameters one peak fit

    fitting_parameters_scaled = scaler.fit_transform(fitting_parameters_one_peak[:, 1:])

    new_dimensions = loss_of_information(fitting_parameters_scaled, 50, f"{survey}_one-peak_parameters_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_fitting_parameters = dimensionality_reduction(fitting_parameters_scaled, new_dimensions)

    best_number = silhouette_score(low_dimension_fitting_parameters, f"{survey}_one-peak_parameters_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_fitting_parameters)

    plot_PCA_with_clusters(low_dimension_fitting_parameters, sn_labels, kmeans, best_number, [1] * len(number_of_peaks), f"{survey}_one-peak_parameters_in_the_PC_space")

    # %%

    #### low dimension + global parameters

    global_parameters_scaled = scaler.fit_transform(global_parameters_one_peak)

    new_dimensions = loss_of_information(global_parameters_scaled, 50, f"{survey}_light_curve_properties_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_global_parameters = dimensionality_reduction(global_parameters_scaled, new_dimensions)

    best_number = silhouette_score(low_dimension_global_parameters, f"{survey}_light_curve_properties_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_global_parameters)

    print(len(np.where(kmeans.labels_ == 1)[0]))

    plot_PCA_with_clusters(low_dimension_global_parameters, sn_labels, kmeans, best_number, number_of_peaks, f"{survey}_light_curve_properties_in_the_PC_space")

    # %%

    #### low dimension + combine the two

    combination_parameters = np.hstack([fitting_parameters[:, 1:], global_parameters])

    combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    new_dimensions = loss_of_information(combination_parameters_scaled, 50, f"{survey}_combined_dataset_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_combination_parameters = dimensionality_reduction(combination_parameters_scaled, new_dimensions)

    best_number = silhouette_score(low_dimension_combination_parameters, f"{survey}_combined_dataset_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_combination_parameters)

    plot_PCA_with_clusters(low_dimension_combination_parameters, sn_labels, kmeans, best_number, number_of_peaks, f"{survey}_combined_dataset_in_the_PC_space")

    # %%

    #### low dimension + combine the two one-peak fit

    combination_parameters = np.hstack([fitting_parameters_one_peak[:, 1:], global_parameters_one_peak])

    combination_parameters_scaled = scaler.fit_transform(combination_parameters)

    new_dimensions = loss_of_information(combination_parameters_scaled, 50, f"{survey}_combined_one-peak_dataset_in_the_PC_space")
    print("best number of dimensions", new_dimensions)
    low_dimension_combination_parameters = dimensionality_reduction(combination_parameters_scaled, new_dimensions)

    best_number = silhouette_score(low_dimension_combination_parameters, f"{survey}_combined_one-peak_dataset_in_the_PC_space")

    kmeans = KMeans(n_clusters = best_number, random_state = 2804)
    kmeans.fit(low_dimension_combination_parameters)

    plot_PCA_with_clusters(low_dimension_combination_parameters, sn_labels, kmeans, best_number, [1] * len(number_of_peaks), f"{survey}_combined_one-peak_dataset_in_the_PC_space")

    # %%

    sample_times_f1, sample_fluxes_f1, sample_times_f2, sample_fluxes_f2 = load_best_fit_light_curves(survey, fitting_parameters_one_peak, [1] * len(number_of_peaks))
    plot_light_curve_template(kmeans, best_number, sample_times_f1, sample_fluxes_f1, f1, f"{survey}_combined_one-peak_dataset_in_the_PC_space")

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
        
    print(sn_labels[cluster_0])
    print(sn_labels[cluster_1])
    print(sn_labels[cluster_2])
