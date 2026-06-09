
"""
Class for storing information about connections between hosts on a network

"""

class Session:
    def __init__(self, time, src_ip, dst_ip, protocol, src_port, dst_port,
                  bytes_sent, bytes_recvd, flow_packets, flow_bytes,
                    avg_pakt_size, max_pakt_size, total_packets, total_payload):
        self.initiated_time = time
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.protocol = protocol
        self.src_port = src_port
        self.dst_port = dst_port
        self.bytes_sent = bytes_sent
        self.bytes_recvd = bytes_recvd
        self.flow_packets = flow_packets
        self.flow_bytes = flow_bytes
        self.avg_pakt_size = avg_pakt_size
        self.max_pakt_size = max_pakt_size
        self.total_packets = total_packets
        self.total_payload = total_payload


