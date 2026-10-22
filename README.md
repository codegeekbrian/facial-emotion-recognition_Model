# Facial Emotion Recognition (CNN)

A Keras/TensorFlow convolutional neural network that classifies 48×48 grayscale face images into seven emotions. It reaches **91.07% accuracy on a held-out test set of 4,165 images**. This is the emotion model used in [MoodTune](https://github.com/codegeekbrian/moodtune-showcase), a music recommender that reads your facial expression.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)

## Results

| Metric | Value |
| --- | --- |
| Test accuracy | **91.07%** |
| Test loss | 0.2597 |
| Test images (never seen during training) | 4,165 |

## Emotion classes

`angry` · `disgust` · `fear` · `happy` · `neutral` · `sad` · `surprise`

## Dataset

A balanced, grayscale version of the RAF-DB face-emotion dataset, hosted on Kaggle and already split into three folders:

| Split | Images |
| --- | --- |
| Train | 30,023 |
| Validation | 7,504 |
| Test | 4,165 |

All images are resized to 48×48 and scaled to the 0–1 range.

## Model

A three-block CNN built with Keras `Sequential`:

| Stage | Layers |
| --- | --- |
| Block 1 | Conv2D (64) → BatchNorm → MaxPool → Dropout 0.25 |
| Block 2 | Conv2D (128) → BatchNorm → MaxPool → Dropout 0.25 |
| Block 3 | Conv2D (256) → BatchNorm → MaxPool → Dropout 0.25 |
| Classifier | Flatten → Dense 512 → BatchNorm → Dropout 0.4 → Dense 256 → Dropout 0.3 → Dense 7 (softmax) |

## Training setup

- **Optimiser:** Adam (learning rate 1e-3), categorical cross-entropy loss, batch size 64
- **Augmentation:** horizontal flip, rotation (10°), zoom (10%), width and height shifts (10%)
- **Callbacks:** model checkpointing on validation accuracy, learning-rate reduction on plateau (×0.5, patience 5), early stopping (patience 15, best weights restored)
- **Evaluation:** the model is chosen using the validation set, and the test set is used once at the end for the final score
- **Hardware:** trained on a Kaggle GPU (NVIDIA T4)

## Output files

Training produces the architecture (`facialemotionmodel.json`), the weights (`facialemotionmodel.weights.h5`), the full saved model (`facialemotionmodel.h5`), and the class order used by the model (`label_classes.json`).

## Limitations

- The model sees small 48×48 grayscale faces, so accuracy on live webcam frames in varied lighting and angles will differ from the test-set score.
- The test set comes from the same dataset as the training data, so it measures performance on similar images, not on any webcam.

## Author

Brian Mutembei · [GitHub](https://github.com/codegeekbrian) · [Portfolio](https://codegeekbrian.page.gd)
