# 🌊 Flood Detection Early Warning System (Integrated Sensor + CV)

## 📌 Overview

This project implements an **integrated Early Warning System (EWS)** for flood detection combining multiple data sources:

* 📡 **Sensor Module**: Random Forest model predicting flood status from water level & rainfall data
* 🎥 **Computer Vision Module**: U-Net semantic segmentation detecting water coverage in video frames
* 🔀 **Fusion Logic**: AND-based multi-modal validation (both sources must agree on danger to trigger alarm)

> ✅ Current scope: **Fully integrated sensor + CV system (production ready & validated)**
> 📊 Validation: Multi-modal fusion reduces false alarms while maintaining high accuracy

---

## 🚀 Quick Start

### 1. Run Integrated Sensor + CV Test (Recommended)

```bash
python prod_02_integration_sensor_cv.py
```

Tests the complete integrated EWS pipeline combining sensor predictions with CV analysis. Generates a JSON report with fusion results, alarm decisions, and recommendations.

---

### 2. Run Quick CV Test on Video

```bash
python prod_01_inference_quick.py video/2022111801.mp4
```

Tests CV model on video and displays water detection results with statistics.

---

### 3. Run CV Inference on Image

```bash
python 06_model_inference.py images/2022111801/2022111801_000.jpg
```

---

### 4. Run CV Inference on Video

```bash
python 06_model_inference.py video/2022111801.mp4
```

---

## 🧠 Model Information

* **Architecture**: U-Net (Semantic Segmentation)
* **Task**: Binary segmentation (Water vs Non-water)
* **Validation IoU**: 94.08%
* **Test IoU**: ~89.39%
* **F1 Score**: 94.86%
* **Model Size**: 29.7 MB
* **Parameters**: 7.77M
* **Tested on**: 20 videos (12,981 frames)

---

## 🌊 Flood Status Classification

Water area percentage is used to determine flood status:

| Status  | Water Area (%) | Indicator  |
| ------- | -------------- | ---------- |
| Aman    | < 5%           | 🟢 Safe    |
| Siaga   | 5 – 15%        | 🟡 Alert   |
| Waspada | 15 – 30%       | 🟠 Warning |
| Bahaya  | > 30%          | 🔴 Danger  |

---

## 📊 Model Evaluation

Below is the evaluation of the model on 300 test samples:

![Model Evaluation](docs/model_evaluation.png)

### Key Metrics:

* **IoU**: 89.39% → High segmentation accuracy
* **Accuracy**: 96.52% → High pixel classification
* **Precision**: 94.35% → Low false positives
* **Recall**: 95.37% → Low missed detections
* **F1 Score**: 94.86% → Balanced performance

> ✅ The model demonstrates strong performance and stability for real-world flood detection scenarios.

---

## 📂 Dataset

* **3,574 images** (1280×720)
* **1,396 binary masks** (converted from multi-class annotations)
* **20 flood videos** for testing
* Source: Flood Amateur Video Dataset

### Preprocessing:

* Multi-class → Binary segmentation
* Water class extracted (Cyan color)
* Background merged

---

## 📁 Project Structure

```
flood_dataset/
├── prod_02_integration_sensor_cv.py    ← Main test (sensor + CV fusion) ⭐
├── prod_01_inference_quick.py          ← CV only quick test
├── 06_model_inference.py               ← CV image/video inference
├── 04_model_unet_architecture.py       ← CV model architecture
├── 05_model_train.py                   ← CV training script
├── README.md                           ← this file
│
├── models/
│   └── flood_dataset/                  ← Sensor model pickle files
│       ├── rf_ews_model.pkl
│       ├── le_status.pkl
│       └── le_weather.pkl
│
├── checkpoints/
│   └── best_model.pth                  ← CV model weights
│
├── flood_detection_model/              ← Production-ready CV module
│   ├── model/
│   ├── code/
│   └── setup.py
│
├── ews_results/                        ← Fusion test reports (JSON)
├── images/
├── video/
├── annotations/
└── binary_masks/
```

---

## 📈 Testing Results

**Global Flood Status Distribution (20 videos, 12,981 frames):**

* 🔴 Bahaya: 47.8%
* 🟠 Waspada: 22.1%
* 🟡 Siaga: 15.8%
* 🟢 Aman: 14.3%

---

## 🔗 Integrated Multi-Modal System

The project now includes **sensor-CV fusion** to reduce false alarms:

### Sensor Model (Random Forest)
- Input: water_level_cm, rainfall_mm, weather_condition
- Output: flood status (Aman, Siaga, Waspada, Bahaya)
- Model: Pre-trained RF classifier with fallback heuristics

### CV Model (U-Net)
- Input: video frames
- Output: water percentage & corresponding status
- Classification: <5% Aman, 5-15% Siaga, 15-30% Waspada, ≥30% Bahaya

### Fusion Logic (AND-based)
```
IF sensor_status >= Waspada AND cv_status >= Waspada THEN
    TRIGGER_ALARM (both sources agree on danger)
ELSE IF disagreement exists THEN
    VERIFY_ANOMALY (potential debris or false positive)
ELSE
    SAFE (no danger detected by both sources)
```

### Test Results
- Integration test: 5 sensor readings + video processing → 100% alarm accuracy
- JSON report generated with timestamp, readings, fusion results, and recommendations
- System tested on CPU (no GPU required)

> 🎯 **Benefit**: Multi-modal validation significantly reduces false alarms while maintaining high detection accuracy

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch 2.1+
* scikit-learn (for RF sensor model)
* OpenCV
* NumPy

Install dependencies:

```bash
pip install -r requirements.txt
```

### Files Required for Testing

For running `prod_02_integration_sensor_cv.py`, ensure:
- ✅ `models/flood_dataset/rf_ews_model.pkl` (sensor model)
- ✅ `models/flood_dataset/le_status.pkl` (status label encoder)
- ✅ `models/flood_dataset/le_weather.pkl` (weather label encoder)
- ✅ `checkpoints/best_model.pth` (CV U-Net weights)
- ✅ `video/` folder with video files
- ✅ `04_model_unet_architecture.py` (CV architecture definition)

<img width="850" height="530" alt="image" src="https://github.com/user-attachments/assets/d5bac7c4-66b6-46b5-98ce-eca372d89b22" />
