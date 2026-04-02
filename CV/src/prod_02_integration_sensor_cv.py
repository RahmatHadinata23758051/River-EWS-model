#!/usr/bin/env python3
"""
INTEGRATED EARLY WARNING SYSTEM (EWS)
Menggabungkan: Sensor Ultrasonik (RF Model) + Computer Vision (U-Net)

WORKFLOW:
1. Load sensor data dari CSV (dummy data)
2. Prediksi dengan Random Forest model → status banjir dari sensor
3. Proses video dengan U-Net → deteksi air dari gambar
4. FUSION: AND Logic - alarm hanya jika KEDUA sumber detect bahaya
5. Generate laporan dengan rekomendasi

Tujuan: Mengurangi false alarm dengan validasi multi-modal
"""

import cv2
import torch
import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import importlib.util
from collections import Counter


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder untuk handle numpy types"""
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class SensorModel:
    """Load Random Forest model untuk prediksi berdasarkan data sensor"""
    
    def __init__(self, model_path, le_status_path, le_weather_path):
        """Inisialisasi RF model"""
        self.model = None
        self.le_status = None
        self.le_weather = None
        self.use_heuristic = False
        
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(le_status_path, 'rb') as f:
                self.le_status = pickle.load(f)
            with open(le_weather_path, 'rb') as f:
                self.le_weather = pickle.load(f)
            print("✓ RF Model (Sensor) berhasil dimuat")
        except Exception as e:
            print(f"⚠️  RF Model gagal: {e}, menggunakan heuristic")
            self.use_heuristic = True
            self._init_heuristic()
    
    def _init_heuristic(self):
        """Init fallback heuristic"""
        from sklearn.preprocessing import LabelEncoder
        self.le_status = LabelEncoder()
        self.le_status.classes_ = np.array(['Aman', 'Siaga', 'Waspada', 'Bahaya'])
        self.le_weather = LabelEncoder()
        self.le_weather.classes_ = np.array(['Cerah', 'Berawan', 'Hujan Ringan', 'Hujan Sedang', 'Hujan Lebat'])
    
    def predict(self, water_level_cm, rainfall_mm, weather):
        """Prediksi status banjir dari data sensor"""
        if self.use_heuristic:
            return self._heuristic_predict(water_level_cm, rainfall_mm, weather)
        
        try:
            weather_encoded = self.le_weather.transform([weather])[0]
            features = np.array([[water_level_cm, rainfall_mm, weather_encoded]])
            pred = self.model.predict(features)[0]
            probs = self.model.predict_proba(features)[0]
            conf = float(probs.max())
            status = self.le_status.inverse_transform([pred])[0]
            return status, conf
        except:
            return self._heuristic_predict(water_level_cm, rainfall_mm, weather)
    
    def _heuristic_predict(self, water_level_cm, rainfall_mm, weather):
        """Heuristic: threshold ketinggian air"""
        if water_level_cm < 50:
            status, conf = 'Aman', 0.95
        elif water_level_cm < 100:
            status, conf = 'Siaga', 0.85
        elif water_level_cm < 150:
            status, conf = 'Waspada', 0.80
        else:
            status, conf = 'Bahaya', 0.90
        
        rainfall_factor = min(rainfall_mm / 100.0, 1.0)
        conf = conf * (0.7 + 0.3 * rainfall_factor)
        return status, float(conf)


class CVFloodDetector:
    """Load U-Net untuk semantic segmentation air"""
    
    def __init__(self, model_path, device='cuda'):
        """Inisialisasi U-Net model"""
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.threshold = 0.5
        self.image_size = 256
        
        try:
            spec = importlib.util.spec_from_file_location(
                "unet_model", 
                Path(__file__).parent / "04_model_unet_architecture.py"
            )
            unet_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(unet_module)
            
            self.model = unet_module.create_model(device=self.device)
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.eval()
            print(f"✓ U-Net Model (CV) berhasil dimuat | Device: {self.device}")
        except Exception as e:
            raise Exception(f"Gagal load U-Net: {e}")
    
    def detect(self, frame):
        """Deteksi air dalam frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        resized = cv2.resize(rgb, (self.image_size, self.image_size))
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            output_np = output.cpu().numpy()[0, 0]
        
        mask = cv2.resize(output_np, (w, h), interpolation=cv2.INTER_LINEAR)
        water_pct = np.sum(mask > self.threshold) / mask.size * 100
        
        if water_pct < 5:
            status = "Aman"
        elif water_pct < 15:
            status = "Siaga"
        elif water_pct < 30:
            status = "Waspada"
        else:
            status = "Bahaya"
        
        confidence = min(mask.max() * 0.95, 1.0) if mask.max() > 0 else 0.5
        
        return {
            'water_pct': water_pct,
            'status': status,
            'confidence': confidence,
            'mask': mask
        }


class SensorCVFusion:
    """Fusi prediksi sensor + CV dengan AND logic"""
    
    @staticmethod
    def status_to_level(status):
        """Konversi status ke level numerik"""
        return {'Aman': 0, 'Siaga': 1, 'Waspada': 2, 'Bahaya': 3}.get(status, 0)
    
    @staticmethod
    def level_to_status(level):
        """Konversi level numerik ke status"""
        return {0: 'Aman', 1: 'Siaga', 2: 'Waspada', 3: 'Bahaya'}.get(level, 'Aman')
    
    @classmethod
    def fuse(cls, sensor_status, sensor_conf, cv_status, cv_conf):
        """
        Fusion dengan AND logic:
        - Trigger alarm hanya jika KEDUA sumber sepakat bahaya
        - Jika disagreement → verifikasi anomali (sampah, etc)
        """
        sensor_level = cls.status_to_level(sensor_status)
        cv_level = cls.status_to_level(cv_status)
        
        # Cek agreement
        agreement = abs(sensor_level - cv_level) <= 1
        
        if agreement:
            fused_level = (sensor_level + cv_level) // 2
            fused_conf = (sensor_conf + cv_conf) / 2
        else:
            fused_level = max(sensor_level, cv_level)
            fused_conf = min(sensor_conf, cv_conf) * 0.7
        
        fused_status = cls.level_to_status(fused_level)
        
        # Decision
        if fused_level >= 2 and agreement:
            decision = 'TRIGGER_ALARM'
            icon = "🚨"
            rec = f"BANJIR TERDETEKSI! Sensor ({sensor_status}) & Kamera ({cv_status}) konfirmasi."
        elif fused_level >= 2 and not agreement:
            decision = 'VERIFY'
            icon = "🔍"
            rec = f"Anomali: Sensor={sensor_status} vs Kamera={cv_status}. Cek sampah/debris."
        elif fused_level >= 1:
            decision = 'MONITOR'
            icon = "👀"
            rec = f"Status {fused_status}: Monitor level air & cuaca."
        else:
            decision = 'SAFE'
            icon = "✓"
            rec = "Aman: Tidak ada bahaya banjir."
        
        return {
            'fused_status': fused_status,
            'decision': decision,
            'confidence': fused_conf,
            'agreement': agreement,
            'recommendation': rec,
            'icon': icon
        }


class IntegratedEWS:
    """Main integrated EWS system"""
    
    def __init__(self):
        """Inisialisasi sistem"""
        print("\n" + "="*80)
        print("INITIALIZING INTEGRATED EWS SYSTEM")
        print("="*80 + "\n")
        
        # Load sensor model (dari folder Ultrasonic)
        sensor_model = Path(__file__).parent.parent.parent / "Ultrasonic" / "models" / "flood_dataset" / "rf_ews_model.pkl"
        le_status = Path(__file__).parent.parent.parent / "Ultrasonic" / "models" / "flood_dataset" / "le_status.pkl"
        le_weather = Path(__file__).parent.parent.parent / "Ultrasonic" / "models" / "flood_dataset" / "le_weather.pkl"
        
        if sensor_model.exists():
            self.sensor = SensorModel(str(sensor_model), str(le_status), str(le_weather))
        else:
            print(f"❌ Sensor model tidak ditemukan: {sensor_model}")
            self.sensor = None
        
        # Load CV model
        cv_model = Path(__file__).parent.parent / "model" / "best_model.pth"
        if cv_model.exists():
            self.cv = CVFloodDetector(str(cv_model))
        else:
            print(f"❌ CV model tidak ditemukan: {cv_model}")
            self.cv = None
        
        if not self.sensor and not self.cv:
            raise Exception("Kedua model tidak tersedia!")
    
    def process_sensor_data(self):
        """Proses data sensor - gunakan dummy data saja"""
        print(f"\n[1] GENERATING DUMMY SENSOR DATA")
        return self._generate_dummy_data()
    
    def _generate_dummy_data(self):
        """Generate dummy sensor data sesuai dengan kondisi video (Bahaya)"""
        data = [
            {'timestamp': '2026-03-27 08:00:00', 'water_level_cm': 140, 'rainfall_mm': 80, 'weather': 'Hujan Sedang'},
            {'timestamp': '2026-03-27 09:00:00', 'water_level_cm': 155, 'rainfall_mm': 95, 'weather': 'Hujan Lebat'},
            {'timestamp': '2026-03-27 10:00:00', 'water_level_cm': 165, 'rainfall_mm': 110, 'weather': 'Hujan Lebat'},
            {'timestamp': '2026-03-27 11:00:00', 'water_level_cm': 180, 'rainfall_mm': 130, 'weather': 'Hujan Lebat'},
            {'timestamp': '2026-03-27 12:00:00', 'water_level_cm': 175, 'rainfall_mm': 125, 'weather': 'Hujan Lebat'},
        ]
        
        print(f"✓ Generated {len(data)} dummy readings (sesuai video condition: HIGH WATER)")
        results = []
        for row in data:
            status, conf = self.sensor.predict(row['water_level_cm'], row['rainfall_mm'], row['weather'])
            results.append({
                'timestamp': row['timestamp'],
                'water_level_cm': row['water_level_cm'],
                'rainfall_mm': row['rainfall_mm'],
                'weather': row['weather'],
                'sensor_status': status,
                'sensor_conf': conf
            })
            print(f"  {row['timestamp']} → {status} (air:{row['water_level_cm']:.1f}cm, hujan:{row['rainfall_mm']:.1f}mm, conf:{conf:.2f})")
        
        return results
    
    def process_video(self):
        """Proses video dengan CV model"""
        print(f"\n[2] PROCESSING VIDEO dengan CV Model")
        
        # Cari video file
        video_dirs = [Path(__file__).parent.parent / "data" / "video", Path("video"), Path("videos")]
        video_file = None
        for video_dir in video_dirs:
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4"))
                if videos:
                    video_file = videos[0]
                    break
        
        if not video_file:
            print("⚠️  Video file tidak ditemukan")
            return None
        
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            print(f"❌ Gagal buka video: {video_file}")
            return None
        
        detections = []
        frame_count = 0
        sample_rate = 10
        
        print(f"  Processing: {video_file.name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % sample_rate == 0:
                det = self.cv.detect(frame)
                det['frame_number'] = frame_count
                detections.append(det)
                
                if len(detections) % 5 == 0:
                    print(f"    Frame #{frame_count}: {det['status']} ({det['water_pct']:.1f}%)")
        
        cap.release()
        
        # Summary
        if detections:
            water_pcts = [d['water_pct'] for d in detections]
            statuses = [d['status'] for d in detections]
            cv_status = Counter(statuses).most_common(1)[0][0]
            cv_conf = float(np.mean([d['confidence'] for d in detections]))
        else:
            cv_status = "Unknown"
            cv_conf = 0.0
        
        print(f"✓ {frame_count} frames processed, {len(detections)} analyzed")
        print(f"  Hasil CV: {cv_status} (conf: {cv_conf:.2f}, avg water: {np.mean(water_pcts):.1f}%)")
        
        return {'cv_status': cv_status, 'cv_conf': cv_conf, 'frames': frame_count}
    
    def fusion_and_report(self, sensor_results, cv_results):
        """Fusion predictions dan generate report"""
        print(f"\n[3] SENSOR-CV FUSION")
        print("="*80)
        
        fusion_results = []
        
        for sr in sensor_results:
            fusion = SensorCVFusion.fuse(
                sr['sensor_status'],
                sr['sensor_conf'],
                cv_results['cv_status'] if cv_results else 'Tidak Ada Data',
                cv_results['cv_conf'] if cv_results else 0.0
            )
            
            result = {
                **sr,
                **fusion,
                'cv_status': cv_results['cv_status'] if cv_results else 'N/A',
                'cv_conf': cv_results['cv_conf'] if cv_results else 0.0
            }
            fusion_results.append(result)
            
            print(f"\n⏰ {sr['timestamp']}")
            print(f"   Sensor : {sr['sensor_status']} (conf={sr['sensor_conf']:.2f})")
            print(f"   CV     : {result['cv_status']} (conf={result['cv_conf']:.2f})")
            print(f"   {fusion['icon']} FUSION: {fusion['fused_status']} | {fusion['decision']}")
            print(f"   → {fusion['recommendation']}")
        
        # Generate final report
        print("\n" + "="*80)
        print("[4] FINAL REPORT")
        print("="*80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': 'Integrated EWS (Sensor + CV Fusion)',
            'sensor_readings': len(sensor_results),
            'cv_frames': cv_results['frames'] if cv_results else 0,
            'fusion_results': fusion_results,
            'alerts': [f for f in fusion_results if f['decision'] in ['TRIGGER_ALARM', 'VERIFY']],
            'safe': [f for f in fusion_results if f['decision'] == 'SAFE']
        }
        
        # Stats
        alert_count = len(report['alerts'])
        safe_count = len(report['safe'])
        
        print(f"\n📊 STATISTIK:")
        print(f"   Total readings: {report['sensor_readings']}")
        print(f"   Alerts: {alert_count}")
        print(f"   Safe: {safe_count}")
        
        if alert_count > 0:
            print(f"\n🚨 CRITICAL ALERTS ({alert_count}):")
            for alert in report['alerts']:
                print(f"   - {alert['timestamp']}: {alert['fused_status']} ({alert['decision']})")
        
        # Save report
        report_dir = Path(__file__).parent.parent / "ews_results"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "integration_test.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, cls=NumpyEncoder)
        
        print(f"\n✓ Report saved: {report_path.absolute()}")
        print("="*80)
    
    def run(self):
        """Run complete integration"""
        
        # Step 1: Sensor
        sensors_ok = self.sensor is not None
        if not sensors_ok:
            print("⚠️  Sensor model tidak tersedia")
            return
        
        sensor_results = self.process_sensor_data()
        
        # Step 2: CV
        cv_ok = self.cv is not None
        cv_results = None
        if cv_ok:
            cv_results = self.process_video()
        else:
            print("⚠️  CV model tidak tersedia, skip video processing")
        
        # Step 3: Fusion
        self.fusion_and_report(sensor_results, cv_results)


def main():
    """Entry point"""
    try:
        ews = IntegratedEWS()
        ews.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
