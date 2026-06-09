import os
from dotenv import load_dotenv
import mysql.connector

load_dotenv()
hostname = os.environ.get("HOST_NAME")
username = os.environ.get("USERNAME")
user_password = os.environ.get("PASSWORD")


def connect_dbclient():
    try:
        # Connect to relational database
        conn = mysql.connector.connect(
                host=hostname,
                user=username,
                password=user_password
                )
        cursor = conn.cursor()

        # Check if database already exists. If not, create it.
        cursor.execute("CREATE DATABASE IF NOT EXISTS rattleDB")
        cursor.execute("USE rattleDB")

        # Create new table for storing data classifications made by Gemini
        cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Classes
                (id INT, name VARCHAR(50) DEFAULT 'Non-threat', label INT DEFAULT 0,
                assessment TEXT, solutions TEXT, options TEXT, time TIMESTAMP NOT NULL,
                CONSTRAINT pk_class PRIMARY KEY (id, name)
                )
                """
                )

        # Create new table for storing network traffic data
        cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Traffic
                (time TIMESTAMP NOT NULL, src_ip VARCHAR(50) NOT NULL, dst_ip VARCHAR(50) NOT NULL,
                protocol VARCHAR(8) NOT NULL, src_port INT NOT NULL, dst_port INT NOT NULL,
                bytes_sent INT DEFAULT 0, bytes_recvd INT DEFAULT 0, flow_packets INT DEFAULT 0,
                flow_bytes INT DEFAULT 0, avg_packet_size INT DEFAULT 0, max_packet_size INT DEFAULT 0,
                num_packets INT DEFAULT 0, payload_size INT DEFAULT 0, id INT, name VARCHAR(50),
                CONSTRAINT fk_class FOREIGN KEY (id, name) REFERENCES Classes(id, name)
                )
                """
                )

        return conn, cursor

    except Exception as error:
        print(f"There is a problem connecting to the database: {error}")
        conn.close()
        cursor.close()

        return None, None
