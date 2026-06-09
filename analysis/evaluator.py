"""
A static class designed to retrieve data from the database for visual representation

"""

import datetime


class Evaluator:
    @staticmethod
    def get_annual_estimates(cursor):
        # Get number of threats discovered in each month of the year
        month_threats = []
        month_safe = []
        try:
            cursor.execute("SELECT label, time FROM Classes")
            items = cursor.fetchall()

            annual_threats = { "January": 0, "February": 0, "March": 0, "April": 0, "May": 0, "June": 0, "July": 0, "August": 0,
                            "September": 0, "October": 0, "November": 0, "December": 0 }
            annual_safe = { "January": 0, "February": 0, "March": 0, "April": 0, "May": 0, "June": 0, "July": 0, "August": 0,
                            "September": 0, "October": 0, "November": 0, "December": 0 }
            
            for i in items:
                label = i[0]
                time = i[1]
                month = time.strftime("%B")
                
                if (label > 0) :
                    annual_threats[month] += 1

                else:
                    annual_safe[month] += 1


            month_threats = list(annual_threats.values())
            month_safe = list(annual_safe.values())

        except Exception as error:
            print(f"There is a problem retrieving data plot: {error}")

        return month_threats, month_safe

    
    @staticmethod
    def get_threat_types(cursor):
        # Get discovered threat types and the count for each one
        types = []
        counts = []
        try:
            cursor.execute("SELECT name FROM Classes")
            output = cursor.fetchall()
            threat_types = {}

            for o in output:
                if (threat_types.get(o[0]) == None):
                    threat_types.update({o[0]: 1})

                else:
                    threat_types[o[0]] += 1

            types = list(threat_types.keys())
            counts = list(threat_types.values())

        except Exception as err:
            print(f"There is a problem retrieving types: {err}")

        return types, counts


    @staticmethod
    def get_traffic_records(cursor):
        # Retrieve packets from database
        cursor.execute("SELECT * FROM Traffic")
        traffic = cursor.fetchall()
        logs = []

        for packet in traffic:
            protocol = "TCP" if packet[3] == "0" else "UDP" if packet[3] == "1" else "ICMP"
            logs.append({
            "Time": packet[0].strftime("%Y-%m-%d %H:%M:%S"),
            "Name": packet[15],
            "Source IP": packet[1],
            "Destination IP": packet[2],
            "Protocol": protocol,
            "Source Port": packet[4],
            "Destination Port": packet[5],
            "Bytes Sent": packet[6],
            "Bytes Received": packet[7],
            "Flow Packets": packet[8],
            "Flow Bytes": packet[9],
            "Average Packet Size": packet[10],
            "Max Packet Size": packet[11],
            "Number of Packets": packet[12],
            "Payload Size": packet[13]
                })

        return logs
