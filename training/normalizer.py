import numpy as np
from datetime import datetime
import ipaddress
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline

"""
A static class designed to preprocess raw data and normalize it for more efficient training.

"""

class Normalizer:
    @staticmethod
    def convert_to_num(map, word):
        if word in map:
            return map[word]
        else: 
            return None

    @staticmethod
    def get_time(time):
        # Computes sine value of the hour in timestamp (hours start at 0)
        dT = datetime.strptime(time, "%m/%d/%Y %H:%M")
        hour = dT.hour
        angle = 2 * np.pi * hour / 24
        feature_sine = np.sin(angle)
        return feature_sine

    @staticmethod
    def estimate_zScores(samples, dPoints):
         # Computes z-scores for each feature
        means = np.mean(samples, axis=0)
        standard_devs = np.std(samples, axis=0)
        z_scores = list(map(lambda d, m, s: (d - m) / s, dPoints, means, standard_devs))
        return z_scores

    @staticmethod
    def encode_data(data):
        # Normalize data to be in numerical format
        protocol_encoding = { "TCP": 0, "UDP": 1, "ICMP": 2 }

        data['Protocol'] = data['Protocol'].map(lambda x: Normalizer.convert_to_num(protocol_encoding, x))
        data['Timestamp'] = data['Timestamp'].map(lambda x: Normalizer.get_time(x))
        data['Source_IP'] = data['Source_IP'].map(lambda x: int(ipaddress.IPv4Address(x)))
        data['Destination_IP'] = data['Destination_IP'].map(lambda x: int(ipaddress.IPv4Address(x)))

        return data
    
    @staticmethod 
    def create_normalization_pipeline():
        # Establish pipeline that scales data and removes redundant features
        scaler = MaxAbsScaler()
        feature_selector = VarianceThreshold(threshold=0.01)
        steps = [('scale_data', scaler), ('feature_select', feature_selector)]
        pipeline = Pipeline(steps)

        return pipeline