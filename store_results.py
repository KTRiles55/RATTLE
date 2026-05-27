import json
import mysql.connector
import datetime
import ipaddress
from dotenv import load_dotenv
import os

load_dotenv()
hostname = os.environ.get("HOST_NAME")
username = os.environ.get("USERNAME")
user_password = os.environ.get("PASSWORD")


def store_results(result, sessions):
    try:
        # Connect to relational database to store new data
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

        # Write results to text file
        output = result.text
        with open("result.txt", "w") as file:
            file.write(output)
            find_json = output.index("json")
            filtered_output = output[find_json:]
            filtered_output = filtered_output.replace("`", "")
            filtered_output = filtered_output.replace("json", "").strip("\n")

        parsed_output = json.loads(filtered_output)

        # Insert new data into tables and update if name and id already exists
        class_query = """
                    INSERT INTO Classes (id, name, label, assessment, solutions, options, time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        label = VALUES(label),
                        assessment = VALUES(assessment),
                        solutions = VALUES(solutions),
                        options = VALUES(options),
                        time = VALUES(time)
                    """
        
        traffic_query = """
                INSERT INTO Traffic (src_ip, dst_ip, protocol, src_port, dst_port, bytes_sent,
                                        bytes_recvd, flow_packets, flow_bytes, avg_packet_size, max_packet_size,
                                        num_packets, payload_size, id, name)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    src_ip = VALUES(src_ip),
                    dst_ip = VALUES(dst_ip),
                    protocol = VALUES(protocol),
                    src_port = VALUES(src_port),
                    dst_port = VALUES(dst_port),
                    bytes_sent = VALUES(bytes_sent),
                    bytes_recvd = VALUES(bytes_recvd),
                    flow_packets = VALUES(flow_packets),
                    flow_bytes = VALUES(flow_bytes),
                    avg_packet_size = VALUES(avg_packet_size),
                    max_packet_size = VALUES(max_packet_size),
                    num_packets = VALUES(num_packets),
                    payload_size = VALUES(payload_size)
                """

        # Check if result is a list of JSON objects
        if not isinstance(parsed_output, list):
            session = sessions[0]
            class_label = 0
            if (parsed_output["Label"] != "Non-threat"):
                class_label = 1

            time = datetime.datetime.now()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            solutions = "No solutions."
            options = "No other options."
            if parsed_output.get("Recommended Solution") != None:
                solutions = parsed_output["Recommended Solution"]["Steps"]
                solutions = "\n".join(solutions)

                if parsed_output.get("Alternative Solution(s)") != None:
                    options = "\n".join(parsed_output["Alternative Solution(s)"])

            values = (parsed_output["Cluster ID"], parsed_output["Label"], class_label, parsed_output["Threat Assessment"], solutions,
                        options, timestamp)

            cursor.execute(class_query, values)

            # Convert ip addresses back from integers to octet notation
            source_ip = str(ipaddress.IPv4Address(session.src_ip))
            dest_ip = str(ipaddress.IPv4Address(session.dst_ip))

            values = (source_ip, dest_ip, session.protocol, session.src_port, session.dst_port, session.bytes_sent, session.bytes_recvd,
                    session.flow_packets, session.flow_bytes, session.avg_pakt_size, session.max_pakt_size, session.total_packets, session.total_payload,
                    parsed_output["Cluster ID"], parsed_output["Label"])
            cursor.execute(traffic_query, values)

        else:
            # Pair each result with its associated network session
            grouping = list(zip(parsed_output, sessions))
            for g in grouping:
                p = g[0]
                s = g[1]

                class_label = 0
                if (p["Label"] != "Non-threat"):
                    class_label = 1

                time = datetime.datetime.now()
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                solutions = "No solutions."
                options = "No other options."
                if p.get("Recommended Solution") != None:
                    solutions = p["Recommended Solution"]["Steps"]
                    solutions = "\n".join(solutions)

                    if p.get("Alternative Solution(s)") != None:
                        options = "\n".join(p["Alternative Solution(s)"])

                values = (p["Cluster ID"], p["Label"], class_label, p["Threat Assessment"], solutions,
                            options, timestamp)

                cursor.execute(class_query, values)
               
                # Convert ip addresses back from integers to octet notation
                source_ip = str(ipaddress.IPv4Address(s.src_ip))
                dest_ip = str(ipaddress.IPv4Address(s.dst_ip))

                values = (source_ip, dest_ip, s.protocol, s.src_port, s.dst_port, s.bytes_sent, s.bytes_recvd,
                        s.flow_packets, s.flow_bytes, s.avg_pakt_size, s.max_pakt_size, s.total_packets, s.total_payload,
                        p["Cluster ID"], p["Label"])
                cursor.execute(traffic_query, values)

        conn.commit()
        print("Successfully added items.")

    except mysql.connector.DatabaseError as derr:
        print(f"\n** Problem with database insertion: {derr}")

    except mysql.connector.Error as err:
        print(f"\n** Caught database server issue: {err}")

    except mysql.connector.InterfaceError as ierr:
        print(f"\n** Caught database interface issue: {ierr}")

    except Exception as err:
        print(f"\n** Caught database exception: {err}")

    finally:
        cursor.close()
        conn.close()
