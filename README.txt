# Multi-QR Code Detection for Medicine Packs
Hackathon Submission: Multi-QR Recognition Challenge

A robust detection system for identifying multiple QR codes on medicine packaging using hybrid computer vision techniques.

## Features

- Stage 1: Multi-QR detection with bounding boxes
- Stage 2: QR decoding and classification (manufacturer, batch, distributor, etc.)
- Handles tilted, blurred, and partially occluded QR codes
- No external APIs - fully self-contained solution
- Multi-method detection pipeline for high accuracy

##  Repository Structure

```
multiqr-hackathon/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train.py                     # Training and validation script
├── infer.py                     # Inference script (main submission)
├── evaluate.py                  # Evaluation script
│
├── src/
│   ├── __init__.py
│   └── detector.py              # Core detection module
│
├── data/                        # Dataset folder (not included)
│   ├── train/                   # Training images
│   ├── test/                    # Test images
│   └── train_annotations.json , test_annotaions.json         # Ground truth annotations
│
└── outputs/
    ├── submission_detection_1.json    # Stage 1 output
    ├── submission_decoding_2.json     # Stage 2 output
```

##  Quick Start

### 1. Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Multiqr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependency (for pyzbar)
# Ubuntu/Debian:
sudo apt-get install libzbar0

# macOS:
brew install zbar
```

### 2. Dataset Setup

Place your dataset in the `data/` folder:

```
data/
├── train/                # Training images
├── test/                 # Test images  
└── annotations.json      # Training annotations
```

### 3. Training/Validation

```bash
# Run validation on training data
python train.py --data_dir data/train --annotations data/train_annotations.json

# Tune parameters
python train.py --data_dir data/train --annotations data/train_annotations.json --tune

# Visualize failures
python train.py --data_dir data/train --annotations data/train_annotations.json --visualize
```

### 4. Inference (Generate Submissions)

#### Stage 1: Detection Only

```bash
python infer.py \
  --input data/test/ \
  --output outputs/submission_detection_1.json
```

Output format:
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {"bbox": [100, 150, 300, 350]},
      {"bbox": [450, 200, 650, 400]}
    ]
  }
]
```

#### Stage 2: Detection + Decoding + Classification

```bash
python infer.py \
  --input data/test_images/ \
  --output outputs/submission_decoding_2.json \
  --decode
```

Output format:
```json
[
  {
    "image_id": "img001",
    "qrs": [
      {
        "bbox": [100, 150, 300, 350],
        "value": "MFR123456",
        "type": "manufacturer"
      },
      {
        "bbox": [450, 200, 650, 400],
        "value": "BATCH789",
        "type": "batch"
      }
    ]
  }
]
```

### 5. Evaluation

```bash
# Evaluate Stage 1 predictions
python evaluate.py \
  --predictions outputs/submission_detection_1.json \
  --ground_truth data/test_annotations.json \
  --stage 1

# Evaluate Stage 2 predictions
python evaluate.py \
  --predictions outputs/submission_decoding_2.json \
  --ground_truth data/test_annotations.json \
  --stage 2
```

##  Technical Approach

### Detection Pipeline

1. **Image Preprocessing** (8 variants):
   - Histogram equalization
   - CLAHE (adaptive)
   - Bilateral filtering
   - Sharpening
   - Morphological operations
   - Otsu's thresholding
   - Adaptive thresholding

2. **Multi-Method Detection**:
   - OpenCV QRCodeDetector
   - WeChat QR detector (if available)
   - pyzbar library
   - Contour-based pattern matching

3. **Post-Processing**:
   - Non-Maximum Suppression (NMS)
   - Duplicate removal
   - Confidence scoring

### Decoding & Classification

- **Decoding**: Multiple attempts with enhanced preprocessing
- **Classification**: Rule-based keyword matching
  - Manufacturer (MFR, MANUF, MAKER)
  - Batch (BATCH, LOT, B#)
  - Distributor (DIST, SUPPLIER)
  - Regulator (REG, FDA, CERT)
  - Serial (SN, SERIAL)
  - Expiry (EXP, DATE, MFG)

##  Performance

Expected metrics on validation set:
- **Precision**: 85-95%
- **Recall**: 70-90%
- **F1 Score**: 75-90%
- **Processing Speed**: ~1-2 seconds per image

## 🛠️ Command-Line Options

### infer.py

```bash
python infer.py --help

Options:
  --input PATH          Input image or folder (required)
  --output PATH         Output JSON file (required)
  --decode              Enable decoding/classification (Stage 2)
  --confidence FLOAT    Confidence threshold (default: 0.5)
```

### train.py

```bash
python train.py --help

Options:
  --data_dir PATH       Training images directory (required)
  --annotations PATH    Annotations JSON file (required)
  --output_dir PATH     Output directory (default: outputs)
  --tune                Tune detection parameters
  --visualize           Visualize failure cases
```

### evaluate.py

```bash
python evaluate.py --help

Options:
  --predictions PATH    Predictions JSON file (required)
  --ground_truth PATH   Ground truth JSON file (required)
  --stage {1,2}         Evaluation stage (default: 1)
  --iou_threshold FLOAT IoU threshold (default: 0.5)
  --output PATH         Save detailed results JSON
```

##  Dependencies

Core requirements:
- Python 3.8+
- OpenCV (opencv-python)
- pyzbar
- NumPy
- tqdm
- matplotlib

See `requirements.txt` for complete list.

## 🔍 Troubleshooting

### Issue: Low recall (missing QR codes)

**Solution**: The detector tries multiple preprocessing methods. If still missing codes:
1. Check image quality (resolution, blur)
2. Try tuning with `--tune` flag
3. Adjust confidence threshold: `--confidence 0.3`

### Issue: pyzbar not working

**Solution**: Install system dependency:
```bash
# Ubuntu/Debian
sudo apt-get install libzbar0

# macOS
brew install zbar

# Then reinstall pyzbar
pip uninstall pyzbar
pip install pyzbar
```

### Issue: WeChat detector not available

**Solution**: This is optional. The system works without it using other methods.

##  How It Works

The system uses a **multi-pass detection strategy**:

1. For each input image, generate 8 preprocessed variants
2. Apply 4 detection methods to each variant (32 total attempts)
3. Collect all candidate detections
4. Remove duplicates using NMS with IoU threshold
5. Optionally decode and classify QR codes

This aggressive multi-method approach achieves high recall while NMS prevents false positives.

##  Submission Files

### submission_detection_1.json (Stage 1)

Required format: List of image predictions with bounding boxes in `[x_min, y_min, x_max, y_max]` format.

### submission_decoding_2.json (Stage 2)

Required format: Same as Stage 1 but with `value` and `type` fields added to each QR.



