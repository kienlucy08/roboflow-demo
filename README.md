# Roboflow + RF-DETR Demo
This repsoitory demonstrates a **minimal end-to-end workflow** for:
1. Authenticating with Roboflow and downloading an object‑detection dataset.
2. Training a RF‑DETR model (DETR‑style transformer) on that data.
3. Running inference and visualising predictions with the supervision library.
Note – This repo is for demonstration/learning purposes only.  Training is done on CPU by default and uses a very small dataset subset so that it can run on any laptop.

---

##  Features

- Connects to a user's Roboflow workspace and lists available projects.
- Adds and loads datasets directly into a selected project.
- Loads and configures an RF-DETR transformer model for training.
- Automatically prepares COCO-style training, validation, and test splits.
- Uses pre-labeled annotations and categories from Roboflow exports.


---

## Project Structure

```bash
roboflow-demo/
│
├── COCO-Dataset-1/         # full dataset download (auto-ignored)
├── My-First-Project-1/     # Roboflow project exports (auto-ignored) version 1
├── My-First-Project-2/     # Roboflow project exports (auto-ignored) version 2
├── limited_dataset/        # subset used for quick testing
├── output/                 # training logs and checkpoints
├── supervision-env/        # local virtual environment
│
├── roboflow_demo.py        # main entry point
├── rf-detr-base.pth        # optional pretrained weights (via Git LFS)
├── requirements.txt
├── .env                    # Roboflow key (not committed)
├── .gitignore
├── README.md
└── .gitattributes          # Git LFS tracking
```

---

## Setup Instructions

### 1. Clone the repo and navigate to it
```bash
git clone https://github.com/kienlucy08/roboflow-demo.git
cd your_file_path/release-notes-gpt
```

### 2 Create / activate a Python 3.10+ virtual‑environment
```bash
python -m venv supervision-env
source supervision-env/bin/activate
```

### 3 Install dependencies (CPU‑only PyTorch)
```bash
pip install -r requirements.txt
```

### 4 Set your Roboflow API key and create your .env file to store it
```bash
.env/
│
├── ROBOFLOW_KEY="<YOUR‑KEY>"
```
This file lives in the `.gitignore` so no need to worry when saving a new change.

### 5 Run the demo – downloads a dataset, trains for 1 epoch, then predicts on a sample image
```bash
python roboflow_demo.py
```

---

## Training Options

| CLI Flag     | Default | Description                                                             |
| ------------ | ------- | ----------------------------------------------------------------------- |
| `epochs`     | `15`    | Number of fine‑tuning epochs.  Reduce for quick tests.                  |
| `batch_size` | `16`    | Effective batch size (will be reduced automatically if running on CPU). |
| `eval`       | `False` | Skip validation to avoid COCO split issues in tiny demos.               |


Edit these directly in roboflow_demo.py or expose them via argparse if you extend the script.

---

## Known Limitations