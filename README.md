# Microplastic Detection with Lightweight Swin Transformer

This project implements a **Lightweight Swin Transformer** for detecting and classifying **microplastics vs algae** in microscopic images.  
It supports data preprocessing, augmentation, model training, evaluation, and inference.

---

## 📂 Project Structure

microplastic-detection/
│
├── data/ # Raw dataset (algae/, microplastics/)
├── data_resized/ # Resized images (output of resize.py)
├── data_augmented/ # Augmented & balanced dataset (output of augment_balance.py)
├── data_preprocessed/ # Final preprocessed dataset (output of preprocess.py)
│
├── src/ # Core source code
│ ├── dataset.py # Dataset loader definitions
│ ├── preprocess.py # Preprocessing & cleaning pipeline
│ ├── augment_balance.py# Augments algae images to balance dataset
│ ├── model.py # Lightweight Swin Transformer model
│ ├── train.py # Training loop
│ ├── metrics.py # Evaluation & metrics reporting
│ ├── detect.py # Detection/visualization on new images
│ ├── optimize.py # Model optimization/export
│ ├── export.py # Export utilities (ONNX, TorchScript)
│
├── results/ # Metrics, confusion matrices, saved predictions
├── classifier.pth # Trained model weights (after train.py)
├── requirements.txt # Python dependencies
└── README.md # Documentation (this file)


---

## ⚙️ Installation

1. Clone repository and create virtual environment:

```bash
git clone <your-repo-url>
cd microplastic-detection
python -m venv .venv
.venv\Scripts\activate   # (Windows PowerShell)
pip install --upgrade pip
pip install -r requirements.txt

## ⚙️ Dataset Preparation
1. Place raw data into 
data/
  ├── algae/
  ├── microplastics/

2. Balance dataset
python src\augment_balance.py --input .\data --outdir .\data_balanced --class algae --target 50 --use_clahe

3.Preprocess
python src/preprocess.py --input "./data_balanced" --outdir "./data_preprocessed" --keep-structure --size 224

4. Train
python train.py --data ./data_preprocessed --epochs 20 --batch 16 --lr 0.0002

5. Metrics & Evaluation
# Full dataset evaluation
python src/metrics.py --data "./data_preprocessed" --weights "classifier.pth" --csv
# Deterministic 80/20 split (recommended for thesis)
python src/metrics.py --data "./data_preprocessed" --weights "classifier.pth" --use_split --csv
#Using the internal validation split
python src/metrics.py --data ./data_preprocessed --weights classifier.pth --use_split --outdir results --csv
#Evaluate on an external test set
python metrics.py --data ./path_to_test_set --weights classifier.pth --outdir results --csv

6. Detection 
# For a single image file:
python src/detect.py --input path/to/pd.jpg --weights classifier.pth --outdir results
# For a directory of images:
python src/detect.py --input path/to/data --weights classifier.pth --outdir results

8. Optimization 
python src/optimize.py --weights "classifier.pth"

7. Export (optional)
python src/export.py


