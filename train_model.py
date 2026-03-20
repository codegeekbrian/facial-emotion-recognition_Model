import glob
import os
import re
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── Google Colab: mount Drive and set working directory ──────────────────────
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE_DIR = '/content/drive/MyDrive/Facial Recogniton Train Model'
    os.chdir(BASE_DIR)
    print("Running on Google Colab. Working directory:", BASE_DIR)
except ImportError:
    print("Not running on Colab. Using local paths.")

# Suppress noisy TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_DIR = 'images/train'
TEST_DIR  = 'images/test'

# ── Build dataframes ─────────────────────────────────────────────────────────
def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        label_dir = os.path.join(dir, label)
        if not os.path.isdir(label_dir):
            continue
        for imagename in os.listdir(label_dir):
            image_paths.append(os.path.join(label_dir, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

test = pd.DataFrame()
test['image'], test['label'] = createdataframe(TEST_DIR)

print("Train shape:", train.shape)
print("Test shape:",  test.shape)

# ── Extract features ──────────────────────────────────────────────────────────
def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, color_mode="grayscale", target_size=(48, 48))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(len(features), 48, 48, 1)
    return features

train_features = extract_features(train['image'])
test_features  = extract_features(test['image'])

x_train = train_features / 255.0
x_test  = test_features  / 255.0

# ── Encode labels ─────────────────────────────────────────────────────────────
le = LabelEncoder()
le.fit(train['label'])
y_train = to_categorical(le.transform(train['label']), num_classes=7)
y_test  = to_categorical(le.transform(test['label']),  num_classes=7)

print("Label classes:", list(le.classes_))

# ── Checkpoint resume logic ───────────────────────────────────────────────────
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.h5"))
start_epoch = 0

if checkpoint_files:
    # Use last-modified time (safe after Google Drive sync)
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    print("Resuming from:", latest_checkpoint)

    model = load_model(latest_checkpoint, compile=False)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    match = re.search(r'facialemotionmodel(\d+)_', os.path.basename(latest_checkpoint))
    if match:
        start_epoch = int(match.group(1))
        print("Resuming from epoch:", start_epoch)

else:
    print("No checkpoint found — building model from scratch")

    # ── Model: 3 Conv blocks only (safe for 48×48 input) ────────────────────
    model = Sequential([
        tf.keras.layers.Input(shape=(48, 48, 1)),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),   # → 24×24
        Dropout(0.25),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),   # → 12×12
        Dropout(0.25),

        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),   # → 6×6
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "facialemotionmodel{epoch:02d}_valacc{val_accuracy:.4f}.h5"),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    x=x_train, y=y_train,
    batch_size=64,
    epochs=100,
    initial_epoch=start_epoch,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_cb, reduce_lr_cb]
)

# ── Save final model ──────────────────────────────────────────────────────────
model_json = model.to_json()
with open("facialemotionmodel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("facialemotionmodel.weights.h5")
model.save("facialemotionmodel.h5")
print("Model saved!")

# ── Plot training history ─────────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print("Training complete.")
