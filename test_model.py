import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img

label = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ── Google Colab: mount Drive and set working directory ──────────────────────
import os
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE_DIR = '/content/drive/MyDrive/Facial Recogniton Train Model'
    os.chdir(BASE_DIR)
    print("Running on Google Colab. Working directory:", BASE_DIR)
except ImportError:
    print("Not running on Colab. Using local paths.")

# ── Load model ────────────────────────────────────────────────────────────────
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.weights.h5")

# ── Preprocess image ──────────────────────────────────────────────────────────
def ef(image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(48, 48))
    feature = np.array(img)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# ── Test example ──────────────────────────────────────────────────────────────
image_path = 'images/train/sad/42.jpg'  # change to any image path
print("Original: sad")
img = ef(image_path)
pred = model.predict(img)
pred_label = label[pred.argmax()]
print("Predicted:", pred_label)

plt.imshow(img.reshape(48, 48), cmap='gray')
plt.title(f"Predicted: {pred_label}")
plt.show()
