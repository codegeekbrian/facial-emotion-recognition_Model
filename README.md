# 🧠 Facial Emotion Recognition — Train Model

A CNN trained to classify facial expressions into **7 emotion categories**:
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

Part of the [MoodTune](https://github.com/) emotion-based music recommender project.

---

## 📁 Project Structure

```
Facial_Recogniton_Train_Model/
├── kaggle_train.ipynb   # ← Main training notebook (run this on Kaggle)
├── train_model.py       # Standalone training script
├── test_model.py        # Run inference on a single image
├── requirements.txt     # Python dependencies
└── images/              # Dataset (not in repo — upload to Kaggle separately)
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        └── (same structure)
```

---

## 🚀 Training on Kaggle (Recommended)

Kaggle provides a free **P100 GPU** (~30 hrs/week) which is significantly faster than a local CPU.

### Step 1 — Upload your Dataset to Kaggle
1. Zip your `images/` folder
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
3. Upload the zip — note the dataset slug (e.g. `facial-emotion-images`)

### Step 2 — Connect this GitHub repo to Kaggle
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. Click **File** → **Link GitHub Account** (one-time setup)
3. After linking: **File** → **Import from GitHub** → select this repo → select `kaggle_train.ipynb`

### Step 3 — Attach the Dataset
- In the notebook, click **Add Data** (right panel) → find your uploaded dataset → Add

### Step 4 — Enable GPU
- **Settings** (right panel) → **Accelerator** → **GPU T4 x2** or **P100**

### Step 5 — Update Dataset Name
In `kaggle_train.ipynb` Step 1 cell, update:
```python
DATASET_NAME = 'your-dataset-slug-here'
```

### Step 6 — Run All & Download Output
- **Run All** → wait for training to complete
- Go to **Output** tab → download `facialemotionmodel.h5`

---

## 🏗️ Model Architecture

| Layer | Details |
|---|---|
| Input | 48×48×1 (grayscale) |
| Conv Block 1 | Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25) → 24×24 |
| Conv Block 2 | Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25) → 12×12 |
| Conv Block 3 | Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25) → 6×6 |
| Dense 1 | Dense(512) + BatchNorm + Dropout(0.4) |
| Dense 2 | Dense(256) + Dropout(0.3) |
| Output | Dense(7, softmax) |

- **Optimizer:** Adam (lr=1e-3, reduces on plateau)
- **Loss:** Categorical Crossentropy
- **Epochs:** 100 · **Batch size:** 64

---

## 🧪 Testing the Model

After downloading `facialemotionmodel.h5` and `facialemotionmodel.json` from Kaggle:

```bash
pip install -r requirements.txt
python test_model.py
```

Edit the `image_path` variable in `test_model.py` to point to any face image.

---

## 📦 Requirements

```
tensorflow
keras
pandas
numpy
tqdm
opencv-contrib-python
scikit-learn
matplotlib
```

Install with:
```bash
pip install -r requirements.txt
```
