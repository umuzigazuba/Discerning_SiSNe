# %%

from data_processing import load_ztf_data, load_atlas_data
from parameter_estimation import light_curve_one_peak, light_curve_two_peaks
from parameter_reduction import determine_parameters, retrieve_redshift, calculate_peak_absolute_magnitude

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True

# %%

survey = "ZTF"
fitting_parameters, global_parameters, number_of_peaks, SN_labels, SN_labels_color = determine_parameters(survey)

scaler = StandardScaler()

# %%

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Define the encoder
input_data = Input(shape=(10,))
encoded = Dense(64, activation='relu')(input_data)
encoded = Dense(32, activation='relu')(encoded)
latent_space = Dense(5, activation='relu')(encoded)

# Define the decoder
decoded = Dense(32, activation='relu')(latent_space)
decoded = Dense(64, activation='relu')(decoded)
output_data = Dense(10, activation='sigmoid')(decoded)  # Output same shape as input

# Define the autoencoder model
autoencoder = Model(input_data, output_data)

# Compile the model
autoencoder.compile(optimizer=Adam(), loss='mse')

# Train the autoencoder
history = autoencoder.fit(train_data, train_data, 
                          epochs=100, 
                          batch_size=32, 
                          validation_data=(val_data, val_data))

encoder_model = Model(input_data, latent_space)
latent_features = encoder_model.predict(data)
