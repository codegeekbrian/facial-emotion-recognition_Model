# Facial Emotion Recognition — Train Model

A CNN that classifies facial expressions into **7 emotion categories**:
`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

---

## Project Structure

```
Facial_Recogniton_Train_Model/
├── kaggle_train.ipynb   # Main training notebook (run on Kaggle)
├── train_model.py       # Standalone training script
├── test_model.py        # Run inference on a single image
├── requirements.txt     # Python dependencies
└── images/
    ├── train/           # Training images per emotion class
    └── test/            # Test images per emotion class
```

---

## Training on Kaggle

### Step 1 — Connect this repo to Kaggle
1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Click **File** → **Link GitHub Account** (one-time setup)
3. After linking: **File** → **Import Notebook from GitHub** → select this repo → select `kaggle_train.ipynb`

### Step 2 — Add the dataset
- In the notebook, click **Add Data** (right panel) → find your dataset → Add
- The dataset will be at `/kaggle/input/<your-dataset-name>/`
- Update `DATASET_NAME` in Step 1 of the notebook to match

### Step 3 — Enable GPU
- **Settings** (right panel) → **Accelerator** → **GPU T4 x2** or **P100**

### Step 4 — Run All & Download Output
- **Run All** → wait for training
- Go to **Output** tab → download `facialemotionmodel.h5`

---

## Model Architecture

| Layer | Details |
|---|---|
| Input | 48×48×1 (grayscale) |
| Conv Block 1 | Conv2D(64) + BatchNorm + MaxPool + Dropout → 24×24 |
| Conv Block 2 | Conv2D(128) + BatchNorm + MaxPool + Dropout → 12×12 |
| Conv Block 3 | Conv2D(256) + BatchNorm + MaxPool + Dropout → 6×6 |
| Dense 1 | Dense(512) + BatchNorm + Dropout(0.4) |
| Dense 2 | Dense(256) + Dropout(0.3) |
| Output | Dense(7, softmax) |

- **Optimizer:** Adam with ReduceLROnPlateau
- **Loss:** Categorical Crossentropy
- **Epochs:** 100 · **Batch size:** 64

---

## Local Testing

After downloading `facialemotionmodel.h5` and `facialemotionmodel.json` from Kaggle output:

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
