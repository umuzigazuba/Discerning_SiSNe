# %%
from kmeans_clustering import plot_PCA, plot_PCA_with_clusters, number_of_clusters

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import random

torch.manual_seed(2804)
np.random.seed(2804)

plt.rcParams["text.usetex"] = True

# %%

survey = "ZTF"
    
fitting_parameters = np.load(f"Data/Input_ML/{survey}/fitting_parameters.npy", allow_pickle = True)
global_parameters = np.load(f"Data/Input_ML/{survey}/global_parameters.npy")
number_of_peaks = np.load(f"Data/Input_ML/{survey}/number_of_peaks.npy")
SN_labels = np.load(f"Data/Input_ML/{survey}/SN_labels.npy")
SN_labels_color = np.load(f"Data/Input_ML/{survey}/SN_labels_color.npy")

scaler = MinMaxScaler()

# %%

class AutoEncoder(nn.Module):

    def __init__(self, n_input, n_nodes):

        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        # self.encoder = nn.Sequential(nn.Dropout(0.2))
        self.encoder = nn.Sequential() 
        self.encoder.append(nn.Linear(n_input, n_nodes[0]))
        self.encoder.append(nn.ReLU())
        for idx in range(len(n_nodes) - 2):
            # self.encoder.append(nn.Dropout(0.2))
            self.encoder.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(n_nodes[-2], n_nodes[-1]))

        self.decoder = nn.Sequential() 
        for idx in range(len(n_nodes) - 1, 0, -1):
            # self.encoder.append(nn.Dropout(0.2))
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

one_peak = np.where(number_of_peaks == 1)

parameters = fitting_parameters[one_peak, 1:15][0].astype(np.float32)
hidden_layers = [8, 16, 2]

parameters_scaled = scaler.fit_transform(parameters)

# Create a TensorDataset with the data as both inputs and targets
tensor_dataset = torch.tensor(parameters_scaled, dtype = torch.float32)
dataset = TensorDataset(tensor_dataset, tensor_dataset)

# Create DataLoader for minibatches
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

length_parameters = len(parameters_scaled[0])
autoencoder = AutoEncoder(n_input = length_parameters, n_nodes = hidden_layers)
autoencoder.train()

# Initialize parameters of AutoEncoder
learning_rate = 1e-3
initialization_loss = nn.MSELoss()
initialization_optimizer = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate) 

epochs = 2000

for epoch in range(epochs):

    for input_parameters, _ in dataloader:
        
        input_parameters = input_parameters.reshape(-1, length_parameters)
        reconstructed = autoencoder(input_parameters)
            
        training_loss = initialization_loss(input_parameters, reconstructed)
    
        initialization_optimizer.zero_grad()
        training_loss.backward()
        initialization_optimizer.step()

latent_representation = autoencoder.encode(tensor_dataset).detach()

latent_representation_names = ["latent_dimension_1", "latent_dimension_2", "latent_dimension_3", "latent_dimension_4", "latent_dimension_5", "latent_dimension_6"]

best_number = number_of_clusters(latent_representation)
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, SN_labels[one_peak], kmeans, best_number, number_of_peaks[one_peak])

# %%

parameters = global_parameters.astype(np.float32)
hidden_layers = [4, 8, 2]

parameters_scaled = scaler.fit_transform(parameters)

# Create a TensorDataset with the data as both inputs and targets
tensor_dataset = torch.tensor(parameters_scaled, dtype = torch.float32)
dataset = TensorDataset(tensor_dataset, tensor_dataset)

# Create DataLoader for minibatches
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)

length_parameters = len(parameters_scaled[0])
autoencoder = AutoEncoder(n_input = length_parameters, n_nodes = hidden_layers)
autoencoder.train()

# Initialize parameters of AutoEncoder
learning_rate = 1e-3
initialization_loss = nn.MSELoss()
initialization_optimizer = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate) 

epochs = 2000

for epoch in range(epochs):

    for input_parameters, _ in dataloader:
        
        input_parameters = input_parameters.reshape(-1, length_parameters)
        reconstructed = autoencoder(input_parameters)
            
        training_loss = initialization_loss(input_parameters, reconstructed)
    
        initialization_optimizer.zero_grad()
        training_loss.backward()
        initialization_optimizer.step()

latent_representation = autoencoder.encode(tensor_dataset).detach()

latent_representation_names = ["latent_dimension_1", "latent_dimension_2", "latent_dimension_3", "latent_dimension_4", "latent_dimension_5", "latent_dimension_6"]

best_number = number_of_clusters(latent_representation)
kmeans = KMeans(n_clusters = best_number, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)

plot_PCA_with_clusters(latent_representation, SN_labels, kmeans, best_number, number_of_peaks)

# %%
# Initialize DEC
DEC = DeepEmbeddedClustering(autoencoder, cluster_centres = None, alpha = 1)

# Initialize cluster centres
latent_dataset = autoencoder.encode(tensor_dataset).detach()

kmeans = KMeans(n_clusters = 2, random_state = 2804)
previous_predictions = kmeans.fit_predict(latent_dataset.numpy())

cluster_centres = torch.tensor(kmeans.cluster_centers_, dtype = torch.float, requires_grad = True)
DEC.cluster_centres = torch.nn.Parameter(cluster_centres)

DEC_loss = nn.KLDivLoss(size_average = False)
learning_rate = 1e-3
DEC_optimizer = torch.optim.SGD(DEC.parameters(), lr = learning_rate, momentum = 0.9)

epochs = 500
tolerance = 10
n_iterations = 0

# for epoch in range(epochs):
while tolerance > 0.001:
    
    for input_parameters, _ in dataloader:
        
        input_parameters = input_parameters.reshape(-1, length_parameters)
        
        latent_parameters = autoencoder.encode(input_parameters)        
        soft_probability = DEC.soft_assignment(latent_parameters)
        target_probability = DEC.target_distribution(soft_probability)
            
        training_loss = DEC_loss(soft_probability.log(), target_probability)
    
        DEC_optimizer.zero_grad()
        training_loss.backward()
        DEC_optimizer.step()

    n_iterations += 1
    
    # Check tolerance
    latent_dataset = autoencoder.encode(tensor_dataset).detach()
    current_predictions = kmeans.fit_predict(latent_dataset.numpy())
    total_changes = np.sum(np.abs(current_predictions - previous_predictions))
    tolerance = total_changes/len(current_predictions)
    print(tolerance)

    previous_predictions = current_predictions

print("number of iterations", n_iterations)

latent_representation = autoencoder.encode(tensor_dataset).detach()

latent_representation_names = ["latent_dimension_1", "latent_dimension_2", "latent_dimension_3", "latent_dimension_4", "latent_dimension_5", "latent_dimension_6"]

kmeans = KMeans(n_clusters = 2, random_state = 2804)
predictions = kmeans.fit_predict(latent_representation)
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

plot_PCA_with_clusters(latent_representation, SN_labels[one_peak], kmeans, 2, number_of_peaks[one_peak])

# %%

cluster_0 = np.where(predictions == 0)
cluster_1 = np.where(predictions == 1)

print(fitting_parameters[one_peak, 0][0][cluster_0])
print(fitting_parameters[one_peak, 0][0][cluster_1])

print(SN_labels[one_peak][cluster_0])
print(SN_labels[one_peak][cluster_1])
# %%
'ZTF23abgnvya' 'ZTF23aaynmrz'
# %%
# %%
