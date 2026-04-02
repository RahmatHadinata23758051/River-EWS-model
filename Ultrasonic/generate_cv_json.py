import json
import random
from datetime import datetime, timedelta

start_time = datetime(2026, 3, 27, 8, 0, 0)
data = []

# Base conditions
water_level = 20.0

# Simulation phases: (count, rainfall_intensity)
# Simulating Normal -> Hujan Deras (banjir) -> Puncak -> Surut -> Total 160
phases = [
    (40, 0),    # Normal
    (40, 4),    # Hujan, air naik
    (30, 8),    # Hujan deras, puncak banjir
    (50, -2)    # Surut
]

current_time = start_time
frame_counter = 1

for count, intensity in phases:
    for _ in range(count):
        # Update Water Level
        if intensity > 0:
            water_level += random.uniform(2.0, 5.0) * intensity
        elif intensity < 0:
            water_level -= random.uniform(3.0, 7.0)
            if water_level < 20.0:
                water_level = random.uniform(20.0, 25.0)
        else:
            water_level += random.uniform(-1.0, 1.0)
            
        # Determine Peil Scale category based on rules
        if water_level < 50:
            peil_scale = "Aman"
        elif water_level < 100:
            peil_scale = "Siaga"
        elif water_level < 150:
            peil_scale = "Waspada"
        else:
            peil_scale = "Bahaya"
            
        # flood_detected is true if Waspada or Bahaya
        flood_detected = peil_scale in ["Waspada", "Bahaya"]
        
        # Confidence score, typical of object detection / segmentation models
        # Confidence might drop slightly during heavy turbulence or rain (which we assume correlates with rapid rises)
        if intensity > 4:
            base_conf = random.uniform(0.70, 0.85) # Lower confidence due to blurring/rain on lens
        else:
            base_conf = random.uniform(0.85, 0.98) # Higher confidence when vision is clear
            
        data.append({
            "timestamp": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "image_url": f"camera/frame_{frame_counter:03d}.jpg",
            "flood_detected": flood_detected,
            "confidence": round(base_conf, 3),
            "peil_scale": peil_scale
        })
        
        current_time += timedelta(minutes=5)
        frame_counter += 1

with open('cv_vision_data.json', 'w') as f:
    json.dump(data, f, indent=4)
    
print("CV dataset generated successfully to cv_vision_data.json!")
