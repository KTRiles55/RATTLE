import datetime

def get_total(cursor):
    num_packets = 0
    num_threats = 0
    num_days = 0
    try:  
        # Get number of packets
        cursor.execute("SELECT * FROM Traffic")
        num_packets = len(cursor.fetchall())

        # Get number of threats
        cursor.execute("SELECT * FROM Classes WHERE name <> 'Non-threat'")
        num_threats = len(cursor.fetchall())
        
        # Get latest time scanned 
        cursor.execute("SELECT MAX(time) FROM Classes")
        latest_time = cursor.fetchall()
        if (latest_time != None):
            timestamp = latest_time[0][0]
                                        
            now = datetime.datetime.now()
            time_difference = abs(now - timestamp)

            num_days = time_difference.days
        else:
            num_days = 0

    except Exception as error:
        print(f"There is a problem with the database retrieval: {error}")

    return num_packets, num_threats, num_days
