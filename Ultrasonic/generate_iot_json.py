import json
import random
from datetime import datetime, timedelta

device_id = "ESP32_US_01"
start_time = datetime(2026, 3, 27, 8, 0, 0)
data = []

# Base conditions
water_level = 25.0
battery = 12.6 # Starts fully charged
signal_base = -65

# Simulation phases: (count, rainfall_intensity)
# Simulating Normal -> Hujan Deras (banjir) -> Puncak -> Surut -> Total 160
phases = [
    (40, 0),    # Normal, kering
    (40, 2),    # Hujan ringan, mulai naik
    (30, 6),    # Hujan sangat deras, naik signifikan
    (50, -1)    # Berhenti hujan, surut perlahan
]

current_time = start_time

for count, intensity in phases:
    for _ in range(count):
        # 1. Update Water Level
        if intensity > 0:
            # Water rises based on intensity + randomness
            water_level += random.uniform(1.0, 3.5) * intensity
        elif intensity < 0:
            # Water recedes naturally
            water_level -= random.uniform(2.0, 5.0)
            if water_level < 20.0:  # Base level
                water_level = random.uniform(20.0, 22.0)
        else:
            # Dry, slight fluctuations
            water_level += random.uniform(-0.5, 0.5)
            
        # Add ultrasonic sensor noise (±0.8 cm is typical for HC-SR04/JSN-SR04T)
        wl_with_noise = water_level + random.uniform(-0.8, 0.8)
        if wl_with_noise < 0: wl_with_noise = random.uniform(0.5, 1.5)

        # 2. Update Battery Voltage
        # Battery drops gradually as device transmits, with tiny sensor voltage reading noise
        battery -= random.uniform(0.005, 0.015)
        if battery < 11.5: 
            battery = 11.5
        batt_with_noise = battery + random.uniform(-0.02, 0.02)
        batt_with_noise = max(11.5, min(12.6, batt_with_noise))

        # 3. Update Signal Strength
        # Signal degrades during heavy rain (higher intensity = worse signal)
        signal = int(random.gauss(signal_base - (intensity * 1.5), 4))
        # Add random dropouts or spikes
        if random.random() < 0.05:
            signal -= random.randint(5, 15)
        signal = max(-90, min(-50, signal))

        data.append({
            "device_id": device_id,
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "water_level_cm": round(wl_with_noise, 1),
            "battery_voltage": round(batt_with_noise, 2),
            "signal_strength": signal
        })
        current_time += timedelta(minutes=5) # Data sent every 5 minutes

with open('iot_sensor_data.json', 'w') as f:
    json.dump(data, f, indent=4)
    
print("IoT dataset generated successfully to iot_sensor_data.json!")
