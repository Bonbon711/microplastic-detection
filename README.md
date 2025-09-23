# Microplastic Detection (Thesis)

Lightweight **ResNet18 + Swin Transformer Tiny** hybrid classifier for **Algae vs Microplastics** detection.  
Pipeline: **Balance → Preprocess → Train → Metrics → Detect → Export → Optimize.

---

## 📂 Project Directory
microplastic-detection/
│
├─ data/ # raw dataset (train/val split or unsplit)
├─ data_balanced/ # balanced dataset (after augment_balance.py)
├─ data_preprocessed/ # preprocessed dataset (resized & normalized)
├─ results/ # metrics reports, confusion matrix, detections
├─ src/ # source code
│ ├─ augment_balance.py
│ ├─ dataset.py
│ ├─ detect.py
│ ├─ export.py
│ ├─ metrics.py
│ ├─ model.py
│ ├─ optimize.py
│ ├─ preprocess.py
│ └─ train.py
├─ requirements.txt
└─ README.md

---

## ⚙️ 1. Environment Setup (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Fix execution policy if activation fails
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

# Balance "algae" class up to 200 images
python src\augment_balance.py --input .\data --outdir .\data_balanced --class algae --target 200 --use_clahe

# Balance "microplastics" class up to 200 images
python src\augment_balance.py --input .\data --outdir .\data_balanced --class microplastics --target 200 --use_clahe

# Resize to 224x224, normalize, output into data_preprocessed/
python src\preprocess.py --input .\data_balanced --output .\data_preprocessed --resize 224

# Auto 80/20 split, 20 epochs, batch size 16, learning rate 2e-4
python src\train.py --data .\data_preprocessed --auto_split --val_ratio 0.2 `
    --epochs 20 --batch_size 16 --lr 2e-4 --output_model classifier.pth

# Generate accuracy, precision, recall, F1 + confusion matrix
python src\metrics.py --data .\data_preprocessed --weights classifier.pth `
    --out .\results --class_names algae microplastics

# Single image detection
python src\detect.py --weights classifier.pth --source .\data\microplastics\1.jpg `
    --output_dir .\results --bbox --cam

# Folder detection
python src\detect.py --weights classifier.pth --source .\data\microplastics `
    --output_dir .\results --bbox --cam

# Export trained model to ONNX
python src\export.py --weights classifier.pth --output model.onnx

# Prune 20% of model weights, save optimized version
python src\optimize.py --weights classifier.pth --output classifier_pruned.pth --prune_fraction 0.2