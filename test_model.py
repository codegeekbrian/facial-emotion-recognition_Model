import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

# ── Load label order ──────────────────────────────────────────────────────────
# Uses label_classes.json if available, otherwise falls back to alphabetical default
if os.path.exists("label_classes.json"):
    with open("label_classes.json") as f:
        labels = json.load(f)
else:
    labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Load model ────────────────────────────────────────────────────────────────
model = load_model("facialemotionmodel.h5")

# ── Preprocess image ──────────────────────────────────────────────────────────
def preprocess(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    img = np.array(img).reshape(1, 48, 48, 1)
    return img / 255.0

# ── Run inference ─────────────────────────────────────────────────────────────
image_path = 'images/train/sad/42.jpg'  # change to any face image
img = preprocess(image_path)
pred = model.predict(img)
pred_label = labels[pred.argmax()]

print("Predicted:", pred_label)

plt.imshow(img.reshape(48, 48), cmap='gray')
plt.title(f"Predicted: {pred_label}")
plt.show()
