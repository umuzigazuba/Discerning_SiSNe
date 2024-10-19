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

    def __init__(self, n_input, n_nodes, cluster_centres, alpha):

        # Initialize the attributes from the parent class (nn.Module)
        super().__init__() 

        self.cluster_centres = cluster_centres
        self.alpha = alpha

        self.encoder = nn.Sequential(nn.Dropout(0.2))
        self.encoder.append(nn.Linear(n_input, n_nodes[0]))
        self.encoder.append(nn.ReLU())
        for idx in range(len(n_nodes) - 2):
            self.encoder.append(nn.Dropout(0.2))
            self.encoder.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(n_nodes[-2], n_nodes[-1]))

        self.decoder = nn.Sequential() 
        for idx in range(len(n_nodes) - 1, 0, -1):
            self.encoder.append(nn.Dropout(0.2))
            self.decoder.append(nn.Linear(n_nodes[idx], n_nodes[idx - 1]))
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(n_nodes[0], n_input))
        self.decoder.append(nn.Sigmoid())
    
    def encode(self, input):

        return self.encoder(input)

    def decode(self, latent):

        return self.decoder(latent)
    
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

        encoded = self.encode(input)
        decoded = self.decode(encoded)
        return decoded

# %%

one_peak = np.where(number_of_peaks == 1)

parameters = fitting_parameters[one_peak, 1:15][0].astype(np.float32)
hidden_layers = [8, 4, 2]

# parameters = global_parameters.astype(np.float32)
# hidden_layers = [4, 2]

parameters_scaled = scaler.fit_transform(parameters)

# Create a TensorDataset with the data as both inputs and targets
tensor_dataset = torch.tensor(parameters_scaled, dtype = torch.float32)
dataset = TensorDataset(tensor_dataset, tensor_dataset)

# Create DataLoader for minibatches
dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

length_parameters = len(parameters_scaled[0])
autoencoder = AutoEncoder(n_input = length_parameters, n_nodes = hidden_layers, cluster_centres = None, alpha = 1)
autoencoder.train()

# Initialize parameters of AutoEncoder
learning_rate = 1e-2
initialization_loss = nn.MSELoss()
initialization_optimizer = torch.optim.Adam(autoencoder.parameters(), lr = learning_rate) 

epochs = 100

for epoch in range(1, epochs):

    for input_parameters, _ in dataloader:
        
        input_parameters = input_parameters.reshape(-1, length_parameters)
        reconstructed = autoencoder(input_parameters)
            
        training_loss = initialization_loss(input_parameters, reconstructed)
    
        initialization_optimizer.zero_grad()
        training_loss.backward()
        initialization_optimizer.step()

# Initialize cluster centres
latent_dataset = autoencoder.encode(tensor_dataset).detach()

kmeans = KMeans(n_clusters = 2, random_state = 2804)
kmeans.fit(latent_dataset.numpy())
cluster_centres = torch.tensor(kmeans.cluster_centers_, dtype = torch.float, requires_grad = True)
autoencoder.cluster_centres = cluster_centres
print(cluster_centres)

DEC_loss = nn.KLDivLoss(size_average = False)
DEC_optimizer = torch.optim.SGD(autoencoder.encoder.parameters(), lr = learning_rate, momentum = 0.9)

epochs = 500

for epoch in range(1, epochs):

    for input_parameters, _ in dataloader:
        
        input_parameters = input_parameters.reshape(-1, length_parameters)
        
        latent_parameters = autoencoder.encode(input_parameters)        
        soft_probability = autoencoder.soft_assignment(latent_parameters)
        target_probability = autoencoder.target_distribution(soft_probability)
            
        training_loss = DEC_loss(soft_probability.log(), target_probability)
    
        DEC_optimizer.zero_grad()
        training_loss.backward()
        DEC_optimizer.step()

    print(autoencoder.cluster_centres)


latent_dataset = autoencoder.encode(tensor_dataset).detach()

kmeans = KMeans(n_clusters = 2, random_state = 2804)
kmeans.fit(latent_dataset.numpy())
cluster_centres = torch.tensor(kmeans.cluster_centers_, dtype = torch.float, requires_grad = True)
print(cluster_centres)
# %%


# %%

# # Defining the Plot Style
# plt.style.use('fivethirtyeight')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
 
# # Plotting the last 100 values
# plt.plot(validation_losses_averaged)

# %%

vae.eval()

with torch.no_grad():
    mean, logvariance = vae.encode(tensor_dataset) 
    latent_representation = vae.reparameterize(mean, logvariance) 

latent_representation = latent_representation.detach().numpy()

latent_representation_names = ["latent_dimension_1", "latent_dimension_2", "latent_dimension_3", "latent_dimension_4", "latent_dimension_5", "latent_dimension_6"]

plot_PCA(latent_representation, SN_labels[one_peak], latent_representation_names[:2])
best_number = number_of_clusters(latent_representation)

kmeans = KMeans(n_clusters = best_number, random_state = 2804)
kmeans.fit(latent_representation)
kmeans.cluster_centers_ = kmeans.cluster_centers_.astype(np.float64)

plot_PCA_with_clusters(latent_representation, SN_labels[one_peak], kmeans, best_number, number_of_peaks[one_peak])

# %%
