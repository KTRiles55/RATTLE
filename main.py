
"""
Author: Kenneth Riles
Date: 2025-07-21

"""

import os
from modeltrainer import ModelTrainer
from normalizer import Normalizer
from validator import ClusterValidator
from scan_pcap import scan_pcap
from store_results import store_results
from fetch_file import fetch_file
from flask import Flask, render_template, request
from packetsniffer import sniff_traffic


def create_platform(trainer, model, bottleneck, dataset):
    app = Flask(__name__)

    # Set directory for files
    app.config['UPLOAD_FOLDER'] = "./uploads"
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Set arguments
    app.config['trainer'] = trainer
    app.config['model'] = model
    app.config['bottleneck'] = bottleneck
    app.config['dataset'] = dataset

    # Retrieve feature details of each cluster using Gemini 2.5 LLM and feed data into the AI
    labels = app.config['trainer'].unfiltered_labels
    filtered_labels = set(labels[labels != -1])

    @app.route("/")
    def load_home():
        return render_template("home.html")

    @app.route("/analysis")
    def load_analysis():
        return render_template("analysis.html")

    @app.route("/records")
    def load_records():
        return render_template("records.html")

    @app.route("/scan-cap", methods=['POST'])
    def scan_capture():
        try:
            # Retrieve file through HTTP POST request
            cap = request.files['capture']
            file_path = fetch_file(cap, app.config['UPLOAD_FOLDER'])

            # Scan pcap file using Gemini AI to make final prediction
            result, sessions = scan_pcap(file_path, app.config['trainer'], app.config['model'], app.config['bottleneck'], app.config['dataset'], filtered_labels)

            # Store generated results and traffic data in MySQL database
            store_results(result, sessions)

            return "Done"

        except Exception as err:
            return f"There is a problem: {err}"

    @app.route("/scan-live", methods=['POST'])
    def scan_live():
        try:
            # Retrieve entered number of packets to scan through HTTP POST request
            limit = request.form.get('count')

            # Scan requested number of packets in real-time
            sniff_traffic(int(limit))

            # Analyze packets captured in newly-written pcap file
            result, sessions = scan_pcap("new_capture.pcap", app.config['trainer'], app.config['model'],
                                         app.config['bottleneck'], app.config['dataset'], filtered_labels)

            # Store generated results and traffic data in MySQL database
            store_results(result, sessions)

            return "Live scan complete."

        except Exception as err:
            return f"Caught live scan error: {err}"
    
    return app


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
    # ** Uncomment the two below statements in the ModelTrainer class source file to
    #    view the elbow method and silhouette score graphs, along with
    #    tested AI-driven cluster analysis **
    trainer.generate_clusters(model, latent_space, 10)

    # Pass arguments into Flask app
    app = create_platform(trainer, model, latent_space, pipeline)

    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
