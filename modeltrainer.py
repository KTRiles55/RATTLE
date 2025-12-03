import os 
import numpy as np
import pandas as pd
import math
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
from tensorflow import data
from keras import optimizers, metrics
from keras.losses import MeanSquaredError
from sklearn.cluster import DBSCAN
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import MaxAbsScaler
from autoencoder import AutoEncoder
from normalizer import Normalizer
from validator import ClusterValidator

SEED = 57
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


"""
Class designated for training the autoencoder model

"""

class ModelTrainer:
    def __init__(self):
        self.features = np.array([])
        self.num_features = 0
        self.dataset = []
        self.training_data = []
        self.batched_training_data = []
        self.testing_data = []
        self.validation_data = []
        self.training_size = 0
        self.labels = []
        self.unfiltered_labels = []


    def split_data(self, file_path, training_amount):
        # Create dataset from csv file
        data = pd.read_csv(file_path)
        self.dataset = pd.DataFrame(data)
        maxRows = self.dataset.iloc[-1].name

        # Encode feature data
        encoded_data = Normalizer.encode_data(data)
        arr = np.array(encoded_data, dtype=np.int64)

        # Split into training and testing data
        train_ratio = int(maxRows * training_amount)
        initial_training_data = arr[:train_ratio]
        self.testing_data = arr[train_ratio:]
    
        # Split training data into final training set and validation data
        init_train_size = len(initial_training_data)
        self.training_data = initial_training_data[:int(init_train_size * 0.80)]
        self.training_size = len(self.training_data)
        self.validation_data = initial_training_data[:int(init_train_size * 0.20)]


    def preprocess_raw_data(self, pipeline):
        # Preprocess data sequentially within a pipeline
        self.training_data = pipeline.fit_transform(self.training_data)
        self.testing_data = pipeline.transform(self.testing_data)
        self.validation_data = pipeline.transform(self.validation_data)

        self.get_features(pipeline)
        return pipeline

    
    def get_features(self, pipeline):
        # Retrieve the total number of selected features and their names
        self.num_features = self.training_data.shape[1]
        selected_features = pipeline.get_feature_names_out()
        feature_vector = [self.dataset.columns[i] for i, v in enumerate(selected_features)]
        self.features = np.append(self.features, feature_vector)


    def batch_data(self, batch_size):
        # Divide training and test data into batches
        shuffled_train = tf.data.Dataset.from_tensor_slices((self.training_data, self.training_data)).shuffle(buffer_size=self.training_size, seed=SEED)
        self.batched_training_data = shuffled_train.batch(batch_size)
        self.testing_data = tf.data.Dataset.from_tensor_slices((self.testing_data, self.testing_data)).batch(batch_size)
        self.validation_data = tf.data.Dataset.from_tensor_slices((self.validation_data, self.validation_data)).batch(batch_size)


    def build_model(self, bottleneck, epochs):
        # Train model using autoencoder neural network
        model = AutoEncoder(bottleneck, self.num_features)
    
        # Compile data with optimization and crossentropy
        model.compile(optimizer=optimizers.Adam(learning_rate=0.06, beta_1=0.9, beta_2=0.95,
              epsilon=1e-4, weight_decay=0.01), loss="mse", 
              metrics=[MeanSquaredError()])

        model.fit(self.batched_training_data, epochs=epochs, validation_data=self.validation_data, verbose=2)

        return model


    def generate_latent_space(self, model):
        # Compress data into a bottleneck for enhanced training performance
        latent_space = model.encoder(self.training_data)

        return latent_space
    

    def generate_clusters(self, model, latent_space, min_samples):
        latent_array = np.array(latent_space)

        """

        # FOR TESTING AND TRAINING PURPOSES: 

        # Find optimal k in elbow method, which will be used to find the optimal epsilon value
        ClusterValidator.plotNeighbors(min_samples, latent_array)
    
        """
    
        # Group dense data in lower dimensional format into clusters
        db = DBSCAN(eps=1.05, min_samples=min_samples)
        db.fit_predict(latent_space)

        # Assign labels from clusters
        labels = db.labels_
        filter_noise = (labels != -1)
        filtered_labels = labels[filter_noise]
        self.labels = set(filtered_labels)
        self.unfiltered_labels = labels
        filtered_space = latent_space[filter_noise]

        """

        # FOR TESTING AND TRAINING PURPOSES: 

        # Calculate and plot silhouette scores to ensure good clustering
        ClusterValidator.plot_silhouette_scores(latent_space, labels, filtered_labels, filtered_space)

        """

        # Return original model dimensionality
        decoded = model.decoder(latent_space)

        """

        # FOR TESTING AND TRAINING PURPOSES:

        # Retrieve feature details of each cluster using Gemini 2.5 LLM
        ClusterValidator.analyze_clusters(latent_space, labels, filtered_labels, self.features)

        """

    def get_predictions(self, model, sessions, latent_space, pipeline):
        # Convert each payload into a latent space
        predictions = np.array([])
        
        for s in sessions: 
            payload = [
                       s.initiated_time, s.src_ip, s.dst_ip, s.protocol, s.src_port, s.dst_port,
                       s.bytes_sent, s.bytes_recvd, s.flow_packets, s.flow_bytes, s.avg_pakt_size,
                       s.max_pakt_size, s.total_packets, s.total_payload
                      ]
            
            # Clean up raw data in new sample through the same normalization pipeline
            payload_arr = np.array(payload, dtype=np.float32)
            expanded_payload = np.array([payload_arr])
            normalized_payload = pipeline.transform(expanded_payload)
            
            # Run new input sample through autoencoder to reduce its dimensionality
            latent_payload = model.encoder(normalized_payload)
            centroids = []

            # Compute distances of new data from centroids of clusters
            for l in self.labels:
                cluster = latent_space[(self.unfiltered_labels == l)]
                centroids.append(np.mean(cluster, axis=0))

            reshaped_centroids = np.vstack(centroids)
            
            # Find distances between new points and centroids to make final predictions
            distances = euclidean_distances(latent_payload, reshaped_centroids)
            cluster_label = np.argmin(distances)
            predictions = np.append(predictions, cluster_label)

        return predictions
        