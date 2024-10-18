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

        self.encoder = nn.Sequential(nn.Linear(n_input, n_nodes[0])) 
        self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Dropout(0.2))
        for idx in range(len(n_nodes) - 2):
            self.encoder.append(nn.Linear(n_nodes[idx], n_nodes[idx + 1]))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(0.2))
        self.encoder.append(nn.Linear(n_nodes[-2], n_nodes[-1]))

        self.decoder = nn.Sequential() 
        for idx in range(len(n_nodes) - 1, 0, -1):
            self.decoder.append(nn.Linear(n_nodes[idx], n_nodes[idx - 1]))
            self.decoder.append(nn.ReLU())
            self.encoder.append(nn.Dropout(0.2))
        self.decoder.append(nn.Linear(n_nodes[0], n_input))
        self.decoder.append(nn.Sigmoid())
    
    def encode(self, input):

        return self.encoder(input)

    def decode(self, latent):

        return self.decoder(latent)

    def forward(self, input):

        latent = self.encode(input)
        decoded = self.decode(latent)
        return decoded, mean, logvariance

def VAE_loss(input, reconstructed, mean, logvariance):
    
    # Reconstruction loss 
    recon_loss = nn.MSELoss()(reconstructed, input)
    
    # KL Divergence loss: D_KL(q(z|x) || p(z)) where q is the approximate and p is N(0, I)
    kl_divergence_loss = -0.5 * torch.sum(1 + logvariance - mean.pow(2) - logvariance.exp())

    # Combine both losses
    return recon_loss + kl_divergence_loss

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
dataloader = DataLoader(dataset, batch_size = 64, shuffle = True)

length_parameters = len(parameters_scaled[0])
vae = VariationalAutoEncoder(n_input = length_parameters, n_nodes = hidden_layers)

learning_rate = 1e-3
# weight_decay = 1e-8 
optimizer = torch.optim.Adam(vae.parameters(), lr = learning_rate) #, weight_decay = weight_decay)

epochs = 500
validation_losses_averaged = []
validation_losses = []
for epoch in range(1, epochs):

    validation_batch = random.randint(0, len(dataloader) - 1)

    for idx, (input_parameters, _) in enumerate(dataloader):
        input_parameters = input_parameters.reshape(-1, length_parameters)

        if idx == validation_batch:
            vae.eval()

            with torch.no_grad():
                reconstructed, mean, logvariance = vae(input_parameters)
                
                validation_loss = VAE_loss(input_parameters, reconstructed, mean, logvariance)
                validation_losses.append(validation_loss.item())

            vae.train()

        else:
            reconstructed, mean, logvariance = vae(input_parameters)
            
            training_loss = VAE_loss(input_parameters, reconstructed, mean, logvariance)
        
            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()

    if epoch % 10 == 0:

        average_validation_loss = np.mean(validation_losses)
        validation_losses_averaged.append(average_validation_loss)
        validation_losses = []
 
# Defining the Plot Style
plt.style.use('fivethirtyeight')
plt.xlabel('Iterations')
plt.ylabel('Loss')
 
# Plotting the last 100 values
plt.plot(validation_losses_averaged)

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
