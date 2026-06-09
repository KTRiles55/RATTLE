import keras as k
import os
from keras import layers
from keras.models import Model

"""
Autoencoder class for anomaly detection that automatically adjusts dimensionality
of input and output layers for better performance.

"""
class AutoEncoder(Model):
    def __init__(self, dimension, n_features):
        super(AutoEncoder, self).__init__()
        self.dimension = dimension
        self.n_features = n_features

        # Lowers dimensionality of model to be processed in the hidden layers
        self.encoder = k.Sequential([
            layers.Input(shape=(self.n_features,)),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'), 
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.dimension)
        ])

        # Reverts model back to original dimension in the output layer
        self.decoder = k.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(units=self.n_features, activation='sigmoid')
        ])

    # Function that implicitly defines encoder and decoder models from input
    def call(self, input_data):
        encoded_model = self.encoder(input_data)
        decoded_model = self.decoder(encoded_model)
        return decoded_model