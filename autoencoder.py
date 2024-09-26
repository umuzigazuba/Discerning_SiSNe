# %%

from data_processing import load_ztf_data, load_atlas_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks
from parameter_reduction import determine_parameters, retrieve_redshift, calculate_peak_absolute_magnitude
from kmeans_clustering import plot_PCA, plot_PCA_with_clusters, number_of_clusters

from sklearn.preprocessing import StandardScaler
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

plt.rcParams["text.usetex"] = True

# %%

survey = "ZTF"
fitting_parameters, global_parameters, number_of_peaks, SN_labels, SN_labels_color = determine_parameters(survey)

# scaler = StandardScaler()
scaler = MinMaxScaler()

# %%

class AutoEncoder(nn.Module):

    def __init__(self, n_input, n_nodes):

        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        self.encoder = nn.Sequential(nn.Linear(n_input, n_nodes[0]), nn.ReLU()) 
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

    def forward(self, input):

        encoded = self.encoder(input)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, input):

        encoded = self.encoder(input)
        return encoded

# %%

# parameters = fitting_parameters[:, 1:].astype(np.float32)
parameters = global_parameters.astype(np.float32)
parameters_scaled = scaler.fit_transform(parameters)

# Create a TensorDataset with the data as both inputs and targets
tensor_dataset = torch.tensor(parameters_scaled, dtype = torch.float32)
dataset = TensorDataset(tensor_dataset, tensor_dataset)

# Create DataLoader for minibatches
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

length_parameters = len(parameters_scaled[0])
hidden_layers = [16, 8, 4]
autoencoder = AutoEncoder(n_input = length_parameters, n_nodes = hidden_layers)

learning_rate = 1e-2
weight_decay = 1e-8 
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate, weight_decay = weight_decay)

epochs = 1000
validation_losses_averaged = []
validation_losses = []
for epoch in range(1, epochs):

    validation_batch = random.randint(0, len(dataloader) - 1)

    for idx, (input_parameters, _) in enumerate(dataloader):
       
        if idx == validation_batch:

            autoencoder.eval()

            with torch.no_grad():
                input_parameters = input_parameters.reshape(-1, length_parameters)
                reconstructed = autoencoder(input_parameters)
                
                validation_loss = loss_function(reconstructed, input_parameters)
                validation_losses.append(validation_loss.item())

            autoencoder.train()

        else:

            input_parameters = input_parameters.reshape(-1, length_parameters)
            reconstructed = autoencoder(input_parameters)
            
            training_loss = loss_function(reconstructed, input_parameters)
        
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

    if epoch % 10 == 0:

        average_validation_loss = np.mean(validation_losses)
        validation_losses_averaged.append(average_validation_loss)
        validation_losses = []
 
# %%

# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
 
# Plotting the last 100 values
plt.plot(validation_losses_averaged)

# %%

latent_representation = autoencoder.encode(tensor_dataset).detach().numpy()

latent_representation_names = ["latent_dimension_1", "latent_dimension_2", "latent_dimension_3", "latent_dimension_4"]

plot_PCA(latent_representation, SN_labels, latent_representation_names)
best_number = number_of_clusters(latent_representation)

kmeans = KMeans(n_clusters = best_number)
kmeans.fit(latent_representation)
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

plot_PCA_with_clusters(latent_representation, SN_labels, kmeans, number_of_peaks)

# %%
kmeans.cluster_centers_
# %%
