import random
from datetime import datetime, timedelta

# Helper function to generate random sensor data
def generate_sensor_data(sensor_id, device_id, sensor_type):
    timestamp = (datetime.now() - timedelta(minutes=random.randint(0, 1000))).isoformat()
    value = random.uniform(20, 40) if sensor_type == "TEMPERATURE" else random.uniform(30, 60)
    return {
        "timestamp": timestamp,
        "id": f"evt_{random.randint(1000, 9999)}",
        "userId": f"usr_{random.randint(1, 5)}",  # Random user
        "assetId": f"asset_{random.randint(100, 999)}",
        "dimension": "VALUE",
        "value": value,
        "originId": sensor_id,
        "originType": sensor_type
    }

# Generate more RDF data dynamically
def generate_rdf_data(num_events=50):
    rdf_data = """
    @prefix ex: <http://example.org/> .
    """
    devices = [f"dev_{i}" for i in range(1, 6)]
    sensors = [f"sens_{i}" for i in range(1, 6)]
    users = [f"usr_{i}" for i in range(1, 6)]
    
    for device in devices:
        rdf_data += f"\nex:{device} ex:hasDeviceName \"Device {device}\" ."
        rdf_data += f"\nex:{device} ex:hasOwner ex:usr_001 ."
    
    for sensor in sensors:
        sensor_type = "TEMPERATURE" if random.random() > 0.5 else "HUMIDITY"
        rdf_data += f"\nex:{sensor} ex:hasSensorType \"{sensor_type}\" ."
        rdf_data += f"\nex:{sensor} ex:hasLocation \"Location {random.randint(1, 3)}\" ."
    
    # Add random events
    for _ in range(num_events):
        sensor_id = random.choice(sensors)
        sensor_type = "TEMPERATURE" if "TEMPERATURE" in sensor_id else "HUMIDITY"
        rdf_data += f"\nex:{generate_sensor_data(sensor_id, random.choice(devices), sensor_type)} ."

    return rdf_data

# Generate and print new RDF data
new_rdf_data = generate_rdf_data(num_events=100)
print(new_rdf_data)
