# 🌊 River EWS — Computer Vision v2.0

Sistem Computer Vision generasi kedua untuk **Early Warning System (EWS) banjir sungai**.
Dibangun di atas dataset RIWA v2 dengan arsitektur DeepLabV3+ untuk segmentasi air yang lebih akurat,
ditambah modul pembacaan level air berbasis peilschaal (staff gauge).

---

## 🗂️ Struktur Project

```
CV-v2/
├── README.md                        ← Dokumen ini
├── requirements.txt                 ← Dependency Phase 1
│
├── Segmentation/                    ← Dataset (sudah ada)
│   └── riwa_v2/
│       ├── images/   (1,142 train images)
│       ├── masks/    (1,142 train masks)
│       ├── test/
│       └── validation/
│
├── phase1_segmentation/             ← 🔵 PHASE 1: Upgrade Segmentasi Air
│   ├── src/
│   │   ├── 01_prepare_dataset.py    ← Verifikasi & prepare dataset
│   │   ├── 02_train_deeplabv3.py    ← Training DeepLabV3+
│   │   └── 03_evaluate.py           ← Evaluasi & perbandingan v1 vs v2
│   └── models/                      ← Hasil training (.pth)
│
├── phase2_gauge/                    ← 🟡 PHASE 2: Deteksi Peilschaal
│   ├── src/
│   │   ├── 01_prepare_gauge_data.py ← Prepare dataset gauge
│   │   ├── 02_train_yolo.py         ← Training YOLOv8
│   │   ├── 03_waterline_detect.py   ← Deteksi waterline di ROI
│   │   └── 04_calibrate.py          ← Kalibrasi pixel → cm
│   ├── data/
│   └── models/
│
└── phase3_integration/              ← 🟢 PHASE 3: Integrasi EWS
    ├── src/
    │   ├── 01_pipeline.py           ← Pipeline utama CV v2
    │   ├── 02_ews_fusion_v2.py      ← Update EWS fusion logic
    │   └── 03_test_endtoend.py      ← End-to-end testing
    └── configs/
        └── calibration.json         ← Kalibrasi pixel→cm (dibuat saat Phase 2)
```

---

## 🚀 Quick Start — Phase 1

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Verifikasi dataset
```bash
cd phase1_segmentation/src
python 01_prepare_dataset.py
```

### 3. Training model
```bash
python 02_train_deeplabv3.py
```

### 4. Evaluasi
```bash
python 03_evaluate.py
```

---

## 📊 Target Performa

| Metric | CV v1 (U-Net) | Target CV v2 |
|--------|--------------|--------------|
| IoU (Test) | 89.39% | > **95%** |
| F1 Score | 94.86% | > **96%** |
| False Positives | Tinggi | < **5%** |
| Model Size | 29.7 MB | < 100 MB |

---

## 📅 Status Pengembangan

- [x] Dataset RIWA v2 tersedia
- [ ] **Phase 1** — Upgrade Water Segmentation ← *Sedang dikerjakan*
- [ ] **Phase 2** — Staff Gauge Detection & Reading
- [ ] **Phase 3** — Integrasi & EWS Output (cm)
