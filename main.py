
"""
Author: Kenneth Riles
Date: 2025-07-21

"""

import os
import json
import time
from training.modeltrainer import ModelTrainer
from training.normalizer import Normalizer
from training.validator import ClusterValidator
from database.connect_dbclient import connect_dbclient
from database.get_total import get_total
from analysis.scan_pcap import scan_pcap
from database.store_results import store_results
from analysis.fetch_file import fetch_file
from flask import Flask, render_template, request
from analysis.packetsniffer import sniff_traffic
from analysis.evaluator import Evaluator


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

    # Connect to database client
    client, cursor = connect_dbclient()

    @app.route("/")
    def load_home():
        return render_template("home.html")

    @app.route("/analysis")
    def load_analysis():
        try:
            n_packets, n_threats, n_days = get_total(cursor)
        
        except Exception as err:
            cursor.close()
            client.close()
            return f"Caught exception: {err}"

        return render_template("analysis.html", packets=n_packets, threats=n_threats, days=n_days)

    @app.route("/records")
    def load_records():
        try:
            monthly_threats, monthly_safe = Evaluator.get_annual_estimates(cursor)
            threat_types, threat_count = Evaluator.get_threat_types(cursor)
            packets = Evaluator.get_traffic_records(cursor)

        except Exception as err:
            cursor.close()
            client.close()
            return f"Caught exception: {err}"

        return render_template("records.html", month_threats=monthly_threats, month_safe=monthly_safe, 
                               types=threat_types, count=threat_count, traffic=packets)

    @app.route("/scan", methods=['POST'])
    def scan_capture():
        try:
            # Retrieve file through HTTP POST request
            result = ""
            method_type = "unknown request"
            start_time = time.time()
            if (request.files.get('capture') != None):
                method_type="File scan"
                cap = request.files['capture']
                file_path = fetch_file(cap, app.config['UPLOAD_FOLDER'])
                
                # Scan pcap file using Gemini AI to make final prediction
                result, sessions = scan_pcap(file_path, app.config['trainer'], app.config['model'], 
                                             app.config['bottleneck'], app.config['dataset'], filtered_labels)

                # Store generated results and traffic data in MySQL database
                store_results(client, cursor, result, sessions)

            elif (request.form.get('count') != None):
                # Retrieve entered number of packets to scan through HTTP POST request
                method_type="Live scan"
                limit = request.form.get('count')
                
                # Scan requested number of packets in real-time
                sniff_traffic(int(limit))

                # Analyze packets captured in newly-written pcap file
                result, sessions = scan_pcap("uploads/new_capture.pcap", app.config['trainer'], app.config['model'],
                                             app.config['bottleneck'], app.config['dataset'], filtered_labels)

                # Store generated results and traffic data in MySQL database
                store_results(client, cursor, result, sessions)
                
            else:
                cursor.close()
                client.close()
                return "Unknown request. Please try again." 

            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)
            
            num_packets = 0
            for s in sessions:
                num_packets += s.total_packets
    
            return render_template("scan.html", output=result, total=num_packets,
                                   method=method_type, time=elapsed_time)
        
        except Exception as err:
            cursor.close()
            client.close()
            return f"Caught {method_type} error: {err}" 

    return app


if __name__ == '__main__': 
    # Setup training model for the dataset
    trainer = ModelTrainer()

    # Split data into training, testing, and validation sets
    trainer.split_data("training/network_traffic.csv", 0.75)

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
