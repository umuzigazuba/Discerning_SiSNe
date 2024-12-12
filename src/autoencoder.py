# %%

__ImportError__ = "One or more required packages are not installed. See requirements.txt."

try:
    from kmeans_clustering import plot_PCA_with_clusters, silhouette_score, load_best_fit_light_curves, plot_light_curve_template, save_clustering_results

    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    import matplotlib.pyplot as plt
    import numpy as np

except ImportError:
    raise ImportError(__ImportError__)

torch.manual_seed(2804)
np.random.seed(2804)

plt.rcParams["text.usetex"] = True

# Colourblind-friendly colours from https://personal.sron.nl/~pault/. 
# Tested using https://davidmathlogic.com/colorblind/
colours = {"blue":"#0077BB", "orange": "#EE7733", "green":"#296529", "purple":"#AA3377", "brown": "#65301A", "cyan": "#33BBEE", "red":"#CC3311"}

# %%

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
sn_labels = np.load(f"../data/machine_learning/{survey}/sn_labels.npy")
sn_labels_color = np.load(f"../data/machine_learning/{survey}/sn_labels_color.npy")

scaler = MinMaxScaler()

# %%

class AutoEncoder(nn.Module):

    """
    Implementation of an autoencoder

    Parameters: 
        n_input (int): Number of nodes in the input layer
        n_nodes (list): List of number of nodes in each hidden layer
    """

    def __init__(self, n_input, n_nodes):

        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        self.encoder = nn.Sequential() 
        self.encoder.append(nn.Linear(n_input, n_nodes[0]))
        self.encoder.append(nn.ReLU())
        for idx in range(len(n_nodes) - 2):
            self.encoder.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(n_nodes[-2], n_nodes[-1]))

        self.decoder = nn.Sequential() 
        for idx in range(len(n_nodes) - 1, 0, -1):
            self.decoder.append(nn.Linear(n_nodes[idx], n_nodes[idx - 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(n_nodes[0], n_input))
        self.decoder.append(nn.Sigmoid())
    
    def encode(self, input):

        return self.encoder(input)

    def decode(self, encoded):

        return self.decoder(encoded)

    def forward(self, input):

        encoded = self.encode(input)
        decoded = self.decode(encoded)
        return decoded

class DeepEmbeddedClustering(nn.Module):

    """
    Implementation of the Deep Embedded Clustering algorithm
    Combines autoencoders and K-Means clustering 

    Parameters: 
        autoencoder (nn.Module): Autoencoder network
        cluster_centres (list): Coordinates of the cluster centres
        alpha (float): Degree's of freedom Student-t distribution
    """

    def __init__(self, autoencoder, cluster_centres, alpha):

        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        self.autoencoder = autoencoder
        self.cluster_centres = cluster_centres
        self.alpha = alpha

    def encode(self, input):

        return self.encoder(input)
    
    def soft_assignment(self, encoded):

        distance = (encoded.unsqueeze(1) - self.cluster_centres.unsqueeze(0)) ** 2
        power = -(self.alpha + 1) / 2

        student_t = (1 + distance / self.alpha) ** power
        probability = student_t / torch.sum(student_t, dim = 1, keepdim = True)

        return probability

    def target_distribution(self, probability):

        normalized_probability = (probability ** 2) / torch.sum(probability, dim = 0)
        target_probability = normalized_probability / torch.sum(normalized_probability, dim = 1, keepdim = True)

        return target_probability

    def forward(self, input):

        encoded = self.autoencoder.encode(input)
        soft_probability = self.soft_assignment(encoded)
        
        return soft_probability 
    
# %% 

def determine_latent_representation(parameter_values, hidden_layers, batch_size, learning_rate, epochs):

    """
    Use an autoencoder to reduce the dimensionality of a parameter space

    Parameters:
        parameter_values (numpy.ndarray): Parameters describing the SNe
        hidden_layers (list): List of number of nodes in each hidden layer
        batch_size (int): Size of batches used to train the autoencoder
        learning_rate (float): Learning rate of the gradient descent method
        epochs (int): Number of epochs the autoencoder is trained for 

    Outputs: 
        latent_representation (numpy.ndarray): Parameters describing the SNe, reduced in dimensionality by the autoencoder
    """

    parameters_scaled = scaler.fit_transform(parameter_values)
    length_parameters = len(parameters_scaled[0])

    # Create a TensorDataset with the data as both inputs and targets
    tensor_dataset = torch.tensor(parameters_scaled, dtype = torch.float32)
    dataset = TensorDataset(tensor_dataset, tensor_dataset)

    # Create DataLoader for minibatches
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    autoencoder = AutoEncoder(n_input = length_parameters, n_nodes = hidden_layers)
    autoencoder.train()

    optimiser = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate) 

    for _ in range(epochs):

        for input_parameters, _ in dataloader:
            
            input_parameters = input_parameters.reshape(-1, length_parameters)
            reconstructed = autoencoder(input_parameters)
                
            training_loss = nn.MSELoss(input_parameters, reconstructed)
        
            optimiser.zero_grad()
            training_loss.backward()
            optimiser.step()

    latent_representation = autoencoder.encode(tensor_dataset).detach()

    return latent_representation

# %%

#### low dimension + fitting parameters

parameter_values = fitting_parameters[:, 1:].astype(np.float32)
hidden_layers = [9, 18, 3]
latent_representation = determine_latent_representation(parameter_values, hidden_layers, 32, 1e-3, 500)

best_number = silhouette_score(latent_representation, f"{survey}_model_fit_parameters_in_the_latent_space")
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, sn_labels, kmeans, best_number, number_of_peaks, f"{survey}_model_fit_parameters_in_the_latent_space")

# %%

#### low dimension + fitting parameters one-peak fit

parameter_values = fitting_parameters_one_peak[:, 1:].astype(np.float32)
hidden_layers = [9, 18, 3]
latent_representation = determine_latent_representation(parameter_values, hidden_layers, 32, 1e-3, 500)

best_number = silhouette_score(latent_representation, f"{survey}_one-peak_model_fit_parameters_in_the_latent_space")
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, sn_labels, kmeans, best_number, [1]*len(number_of_peaks), f"{survey}_one-peak_model_fit_parameters_in_the_latent_space")

# %%

#### low dimension + global parameters

parameter_values = global_parameters.astype(np.float32)
hidden_layers = [9, 21, 3]
latent_representation = determine_latent_representation(parameter_values, hidden_layers, 32, 1e-4, 1000)

best_number = silhouette_score(latent_representation, f"{survey}_light_curve_properties_in_the_latent_space")
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, sn_labels, kmeans, best_number, number_of_peaks, f"{survey}_light_curve_properties_in_the_latent_space")

# %%

#### low dimension + combine the two one-peak fit

parameter_values = np.hstack([fitting_parameters_one_peak[:, 1:], global_parameters_one_peak])
hidden_layers = [21, 36, 3]
latent_representation = determine_latent_representation(parameter_values, hidden_layers, 32, 1e-4, 1000)

best_number = silhouette_score(latent_representation, f"{survey}_combined_one-peak_dataset_in_the_latent_space")
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, sn_labels, kmeans, best_number, [1] * len(number_of_peaks),  f"{survey}_combined_one-peak_dataset_in_the_latent_space")

# %%

sample_times_f1, sample_fluxes_f1, sample_times_f2, sample_fluxes_f2 = load_best_fit_light_curves(survey, fitting_parameters_one_peak, [1] * len(number_of_peaks))
plot_light_curve_template(kmeans, best_number, sample_times_f1, sample_fluxes_f1, f1, f"{survey}_combined_one-peak_dataset_in_the_latent_space")
save_clustering_results(kmeans, best_number, fitting_parameters_one_peak[:, 0], sn_labels, f"{survey}_combined_one-peak_dataset_in_the_latent_space")

# %%

# # Initialize DEC
# DEC = DeepEmbeddedClustering(autoencoder, cluster_centres = None, alpha = 1)

# # Initialize cluster centres
# latent_dataset = autoencoder.encode(tensor_dataset).detach()

# kmeans = KMeans(n_clusters = best_number, random_state = 2804)
# previous_predictions = kmeans.fit_predict(latent_dataset.numpy())

# cluster_centres = torch.tensor(kmeans.cluster_centers_, dtype = torch.float, requires_grad = True)
# DEC.cluster_centres = torch.nn.Parameter(cluster_centres)

# DEC_loss = nn.KLDivLoss(size_average = False)
# learning_rate = 1e-3
# DEC_optimizer = torch.optim.SGD(DEC.parameters(), lr = learning_rate, momentum = 0.9)

# epochs = 500
# tolerance = 10
# n_iterations = 0

# # for epoch in range(epochs):
# while tolerance > 0.001:
    
#     for input_parameters, _ in dataloader:
        
#         input_parameters = input_parameters.reshape(-1, length_parameters)
        
#         latent_parameters = autoencoder.encode(input_parameters)        
#         soft_probability = DEC.soft_assignment(latent_parameters)
#         target_probability = DEC.target_distribution(soft_probability)
            
#         training_loss = DEC_loss(soft_probability.log(), target_probability)
    
#         DEC_optimizer.zero_grad()
#         training_loss.backward()
#         DEC_optimizer.step()

#     n_iterations += 1
    
#     # Check tolerance
#     latent_dataset = autoencoder.encode(tensor_dataset).detach()
#     current_predictions = kmeans.fit_predict(latent_dataset.numpy())
#     total_changes = np.sum(np.abs(current_predictions - previous_predictions))
#     tolerance = total_changes/len(current_predictions)
#     print(tolerance)

#     previous_predictions = current_predictions

# print("number of iterations", n_iterations)

# latent_representation = autoencoder.encode(tensor_dataset).detach()

# kmeans = KMeans(n_clusters = best_number, random_state = 2804)
# predictions = kmeans.fit_predict(latent_representation)
# kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

# plot_PCA_with_clusters(latent_representation, sn_labels, kmeans, 2, [1] * len(number_of_peaks), f"{survey}_SNe_using_an_autoencoder")

