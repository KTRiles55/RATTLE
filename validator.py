import os
import numpy as np
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from normalizer import Normalizer

load_dotenv()
credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

"""
A static class designed to validate clusters and labeling accuracy

"""

class ClusterValidator:
    @staticmethod 
    def findNeighbors(total_samples, latent_space):
        # Calculate the euclidean distance for each sample to identify k-nearest neighbors
        k=total_samples-1
        neighbors = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(latent_space)
        
        # Calculate the mean for distances between each sample and their neighbors
        distances, _ = neighbors.kneighbors(latent_space)
        dist_means = np.mean(distances[:, 1:k], axis=1)
        k_distances = np.sort(dist_means, axis=0)[::-1]

        return k_distances

    @staticmethod
    def plotNeighbors(total_samples, latent_space):
        # Plot the distances of k-nearest neighbors on a logistic curve graph to support elbow method
        neighbors = ClusterValidator.findNeighbors(total_samples, latent_space)
        fig, ax = plt.subplots()
        ax.plot(neighbors, marker='*')

        # Set x and y axis ranges
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))

        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(0.5))
        plt.show()

    @staticmethod
    def plot_silhouette_scores(latent_space, labels, f_labels, f_space):
        # Calculate and plot silhouette scores for each sample in the lower dimensional field
        scores = silhouette_samples(latent_space, labels)
        plt.plot(scores, marker='*')
        plt.show()
    
        plt.scatter(f_space[:, 0], f_space[:, 1], c=f_labels, cmap='viridis')
        plt.show()

    @staticmethod
    def analyze_clusters(latent_space, labels, f_labels, feature_names):
        # Analyze cluster characteristics using Gemini LLM to assign possible labels
        unique_labels = set(f_labels)
        cluster_arr = np.array([])

        vertexai.init(project="seismic-lexicon-474904-g9", location="us-central1")
        model = GenerativeModel("gemini-2.5-pro")
        system_prompt = (
            "You are a cybersecurity expert analyzing clusters of network traffic. "
            "Each cluster contains similar behavior. For each cluster, you must:\n\n"
            "1. Describe the behavior based on statistical summaries of features.\n"
            "2. Identify if the behavior represents a threat (e.g., DDoS, Scanning, Malware, Data Exfiltration, or any other type of cyber threat). "
            "You might even encounter a new type of threat that has not been identified yet.\n"
            "3. Assign a label: either 'Non-threat' or a specific threat type.\n\n"
            "The feature values are normalized (e.g., z-scores). For comparison, the baseline cluster has z-scores between -1 and 1. "
            "Values above 1 are significantly higher than average; values below -1 are significantly lower. "
            "Use this context to interpret whether a feature's behavior is abnormal.\n\n"
            "4. Suggest a step-by-step solution that a cybersecurity analyst should follow to respond to this threat and resolve it. "
            "Explain each step clearly so that it can be understood regardless of the reader’s technical background. "
            "Also explain *why* this approach is effective, and provide an estimated probability of success rate (as a percentile range). "
            "Feel free to suggest alternative solutions that might offer a higher probability of success rate or additional protection. "
            "Respond in the following format:\n"
            "{\n\"Cluster ID\": <int>\n"
            "\"Behavior Description\":\n\n"
            "... \n"
            "\"Threat Assessment\":\n\n"
            "... \n"
            "\"Label\": <Non-threat | Threat Type>\n\n"
            "\"Recommended Solution\":\n\n"
            "... (step-by-step solution, explanation, and success rate)\n"
            "// Only include this field if the \"Label\" is a Threat Type\n"
            "\"Alternative Solution(s)\":\n\n"
            "\"1. ...\",\n"
            "\"2. ...\"\n"
            "// Only include this field if there are valid alternatives\n}"
        ) 
    
        for i, l in enumerate(unique_labels):
            # Retrieve samples from the cluster with the assigned label
            c_samples = latent_space[(labels == l)]
            num_samples = len(c_samples)

            # Compute the z-scores for each data point in the cluster
            z_score_matrix = [Normalizer.estimate_zScores(c_samples, p) for p in c_samples]

            # Find which set of z_scores in cluster represents the average
            average_score = np.mean(z_score_matrix)
            min_difference = 99
            avg_score_index = -1
            for index, scores in enumerate(z_score_matrix):
                sample_mean = np.mean(scores)
                score_difference = abs(average_score - sample_mean)
                if (score_difference < min_difference):
                    min_difference = score_difference
                    avg_score_index = index
            mean_features = np.array(z_score_matrix[avg_score_index])

            # Find the most prominent features based on their z-score
            dom_feature_indices = np.argsort(mean_features)[::-1][:5]
            features = [feature_names[i] for i in dom_feature_indices]
            feature_values = mean_features[dom_feature_indices]
            feature_summary = f"Dominant Features: {features}\nFeature Scores: {feature_values}"
            cluster_arr = np.append(cluster_arr, {"Cluster ID": l, "Number of Samples": num_samples, "Feature Summary": feature_summary })

            user_prompt = (
                f"Cluster ID: {cluster_arr[i]['Cluster ID']}\n"
                f"Number of Samples: {cluster_arr[i]['Number of Samples']}\n"
                f"Feature Summaries:\n{cluster_arr[i]['Feature Summary']}\n"
            )
            response = model.generate_content(system_prompt + "\n\n" + user_prompt)
            print(response.text)