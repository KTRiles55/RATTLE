from scapy.all import *
from collections import defaultdict
from session import Session
import numpy as np
from datetime import datetime
from normalizer import Normalizer
import ipaddress

def retrieve_pcap_content(file_path): 
    # Read network traffic from pcap file saved from Wireshark
    capture = rdpcap(file_path)
    return capture

def recordTraffic(packets):
    conversations = defaultdict(list)
    # Find each pair of ip addresses from all packets
    for packet in packets:
        if packet.haslayer(IP):
            source_addr = packet[IP].src
            dest_addr = packet[IP].dst
            key = tuple(sorted([source_addr, dest_addr]))

            # Record traffic between each pairing
            conversations[key].append(packet)
    return conversations

def retrieve_payloads(conversations):
    # Examine payload data of each conversation between each pair of ip addresses
    sessions = np.array([])
    for clients, data in conversations.items():
        source_address = clients[0]
        dest_address = clients[1]
        sent_bytes = 0
        received_bytes = 0
        flow_packets = 0
        protocol = -1
        source_port = 0
        dest_port = 0

        # Calculate total time taken to send all packets
        start_time = data[0].time
        end_time = data[-1].time
        elapsed_time = end_time - start_time

        # Calculate number of flow packets per second
        total_packets = len(data)
        if elapsed_time > 0:
            flow_packets = total_packets / elapsed_time
        else:
            flow_packets = total_packets

        # Inspect raw binary data for total bytes sent and received at a specific ip address
        for d in data:
            if d.haslayer(Raw):
                if d[IP].src == source_address:
                    sent_bytes += len(d[Raw].load)

                elif d[IP].dst == source_address:
                    received_bytes += len(d[Raw].load)

            if d.haslayer(TCP):
                protocol = 0
            elif d.haslayer(UDP):
                protocol = 1
            elif d.haslayer(ICMP):
                protocol = 2

            source_port = d[IP].sport
            dest_port = d[IP].dport

        # Calculate the number of flow bytes per second
        total_payload = sent_bytes + received_bytes
        if elapsed_time > 0:
            flow_bytes = total_payload / elapsed_time
        else:
            flow_bytes = total_payload
    
        # Estimate the average packet size and find the maximum length
        average_pkt_size = total_payload / total_packets
        max_pkt_size = max([len(d) for d in data])

        # Get timestamp from time of first packet sent in session and encode
        session_date = datetime.fromtimestamp(int(start_time))
        session_timestamp = session_date.strftime("%m/%d/%Y %H:%M")
        encoded_timestamp = Normalizer.get_time(session_timestamp)

        # Encode ip addresses in numerical form 
        encoded_src_ip = int(ipaddress.IPv4Address(source_address))
        encoded_dst_ip = int(ipaddress.IPv4Address(dest_address))

        sessions = np.append(sessions, Session(encoded_timestamp, encoded_src_ip, encoded_dst_ip, protocol,
                               source_port, dest_port, sent_bytes, received_bytes,
                               flow_packets, flow_bytes, average_pkt_size, max_pkt_size,
                               total_packets, total_payload))

    return sessions
    

    
        