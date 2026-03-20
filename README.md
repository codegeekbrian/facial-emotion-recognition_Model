# Facial Emotion Recognition — Train Model

I built a custom CNN to classify facial expressions into **7 emotion categories**:
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

I trained this model using Kaggle's free GPU (T4 x2) with the [nishchalchandel/raf-db-face-emotion-dataset](https://www.kaggle.com/datasets/nishchalchandel/raf-db-face-emotion-dataset), achieving **78.2% validation accuracy**.

---

## Project Structure

```
Facial_Recogniton_Train_Model/
├── kaggle_train.ipynb   # Main training notebook — run this on Kaggle
├── test_model.py        # Run inference on a single image locally
└── requirements.txt     # Python dependencies
```

---

## How I Train on Kaggle

I use Kaggle's free GPU instead of training locally because it's significantly faster and I can close my browser while it runs.

### Step 1 — Connect this repo to Kaggle
1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Click **File** → **Link GitHub Account** (one-time setup)
3. After linking: **File** → **Import Notebook from GitHub** → select this repo → select `kaggle_train.ipynb`

### Step 2 — Add the dataset
- Right panel → **Add Data** → search `nishchalchandel raf-db-face-emotion-dataset` → Add
- It mounts automatically at `/kaggle/input/raf-db-face-emotion-dataset/`

### Step 3 — Enable GPU
- Right panel → **Session options** → **Accelerator** → **GPU T4 x2**

### Step 4 — Train offline
- Click **Save Version** → **Save & Run All (Commit)**
- I can close my browser — Kaggle keeps running in the background
- When done, go to the **Output** tab → download `facialemotionmodel.h5` and `label_classes.json`

---

## Model Architecture

I settled on a 3-block CNN after testing several configurations. It consistently outperformed deeper or larger variants on this dataset.

| Layer | Details |
|---|---|
| Input | 48×48×1 (grayscale) |
| Conv Block 1 | Conv2D(64) + BatchNorm + MaxPool + Dropout(0.25) → 24×24 |
| Conv Block 2 | Conv2D(128) + BatchNorm + MaxPool + Dropout(0.25) → 12×12 |
| Conv Block 3 | Conv2D(256) + BatchNorm + MaxPool + Dropout(0.25) → 6×6 |
| Dense 1 | Dense(512) + BatchNorm + Dropout(0.4) |
| Dense 2 | Dense(256) + Dropout(0.3) |
| Output | Dense(7, softmax) |

- **Optimizer:** Adam (lr=1e-3) with ReduceLROnPlateau
- **Loss:** Categorical Crossentropy
- **Epochs:** 100 (with EarlyStopping, patience=15)
- **Batch size:** 64
- **Augmentation:** horizontal flip, rotation ±10°, zoom ±10%, shift ±10%

---

## Testing the Model Locally

After downloading `facialemotionmodel.h5` and `label_classes.json` from Kaggle:

```bash
pip install -r requirements.txt
python test_model.py
```

Edit `image_path` in `test_model.py` to point to any face image.

---

## Requirements

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
