"""
Author: Kenneth Riles
Date: 2025-07-21

"""

from modeltrainer import ModelTrainer
from packetsniffer import *
from session import Session
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def load_home():
    return render_template("home.html")

@app.route("/analysis")
def load_analysis():
    return render_template("analysis.html")

if __name__ == '__main__': 
    # Setup training model for the dataset
    trainer = ModelTrainer()

    # Split data into training, testing, and validation sets
    trainer.split_data('network_traffic.csv', 0.75)

    # Clean up and transform data into usable format for efficient model training
    pipeline = Normalizer.create_normalization_pipeline()
    trainer.preprocess_raw_data(pipeline)

    # Segment data into 16-bit batches to increase processing speed of large numbers of samples
    trainer.batch_data(16)

    # Set up autoencoder model and compress its bottleneck to 16 input neurons
    # Cycle through 5 epochs until the neural network reaches an early stop to
    # reduce overfitting on training model when validation loss begins increasing
    model = trainer.build_model(6, 5)

    # Get lower dimensional representation of data
    latent_space = trainer.generate_latent_space(model)

    # Identify new labels by clustering the data based on their density regions
    # ** Uncomment the two below statements in the Trainer class source file to
    #    view the elbow method and silhouette score graphs, along with
    #    tested AI-driven cluster analysis **
    trainer.generate_clusters(model, latent_space, 10)
    
    # Retrieve network traffic data for prediction
    capture = retrieve_pcap_content(" --INSERT PCAP FILE HERE-- ")
    traffic = recordTraffic(capture)
    sessions = retrieve_payloads(traffic)

    predictions = trainer.get_predictions(model, sessions, latent_space, pipeline)
    
    """
    Work in progress...

    """


    # Run Flask server
    app.run()