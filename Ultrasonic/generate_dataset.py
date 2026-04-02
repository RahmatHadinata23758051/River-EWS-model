import csv
import random
import math
from datetime import datetime, timedelta

# Requirements:
# timestamp, water_level_cm, rainfall_mm, weather_condition, ai_visual, peil_scale, status
# Aman (<50), Siaga (50–100), Waspada (100–150), Bahaya (>150)

def get_status(wl):
    if wl < 50: return "Aman"
    if wl < 100: return "Siaga"
    if wl < 150: return "Waspada"
    return "Bahaya"

def get_weather(rf):
    if rf == 0: return "Cerah"
    if rf < 5: return "Berawan"
    if rf < 20: return "Hujan Ringan"
    if rf < 50: return "Hujan Sedang"
    return "Hujan Lebat"

def get_ai_visual(wl, rf, trend):
    if wl > 150: return random.choice(["Area Tergenang Parah", "Banyak Sampah/Debris", "Air Keruh Meluap"])
    if wl > 100: return random.choice(["Air Tinggi", "Arus Deras", "Sedikit Debris"])
    if wl > 50 and trend > 0: return "Air Naik Cepat"
    if rf > 10: return "Kamera Terkena Hujan"
    return "Normal / Clear"

rows = []
start_time = datetime(2026, 3, 27, 8, 0, 0)
water_level = 20.0 # start at 20cm

# Simulation phases: Normal (40), Hujan Mulai (40), Puncak (30), Surut (50) -> Total 160
phases = [
    (40, 0, 0),         # Normal: 40 rows, 0 avg rain
    (40, 15, 30),       # Hujan Mulai: 40 rows, 15-30 mm rain
    (30, 50, 80),       # Puncak Hujan: 30 rows, 50-80 mm rain
    (50, 0, 5)          # Surut: 50 rows, 0-5 mm rain
]

current_time = start_time
for phase in phases:
    count, min_rf, max_rf = phase
    for _ in range(count):
        rainfall = random.uniform(min_rf, max_rf)
        if rainfall == 0 and min_rf == 0 and random.random() < 0.8:
            rainfall = 0.0
            
        # Rainfall affects water level
        base_increase = rainfall * 0.4
        
        # Natural drainage/surut
        drainage = 3.0 if water_level > 50 else 1.0
        if water_level > 100: drainage = 5.0
        if water_level > 150: drainage = 8.0
        
        # Noise
        noise = random.uniform(-1.5, 2.5)
        
        prev_wl = water_level
        water_level = water_level + base_increase - drainage + noise
        if water_level < 10: water_level = random.uniform(10, 15)
        
        trend = water_level - prev_wl
        
        rows.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "water_level_cm": round(water_level, 1),
            "rainfall_mm": round(rainfall, 1),
            "weather_condition": get_weather(rainfall),
            "ai_visual": get_ai_visual(water_level, rainfall, trend),
            "peil_scale": round(water_level),
            "status": get_status(water_level)
        })
        current_time += timedelta(minutes=15)

with open('flood_dataset.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["timestamp", "water_level_cm", "rainfall_mm", "weather_condition", "ai_visual", "peil_scale", "status"])
    writer.writeheader()
    writer.writerows(rows)
    print("Dataset generated successfully!")
