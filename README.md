# Facial-Emotion-Detection-Using-CNN
A deep learning-based Facial Emotion Detection System developed using TensorFlow/Keras, OpenCV, and the FER2013 dataset. The model classifies facial expressions into seven different emotions and performs real-time emotion detection using a webcam.

## Project Overview

A deep learning-based Facial Emotion Detection System developed using **TensorFlow/Keras**, **OpenCV**, and the **FER2013 dataset**. The model classifies facial expressions into seven different emotions and performs real-time emotion detection using a webcam.

---

## 📌 Features

* Facial Emotion Recognition using CNN
* Trained on the FER2013 dataset
* Image preprocessing and normalization
* Data Augmentation
* Batch Normalization
* Dropout Regularization
* Early Stopping
* Accuracy & Loss Visualization
* Classification Report
* Confusion Matrix
* Real-Time Webcam Emotion Detection using OpenCV
* Model saved as `emotion_model.h5`

---

## 📂 Project Structure

```
Emotion-Detection/
│
├── emotion.ipynb              # CNN Training Notebook
├── apps.py                    # Real-Time Emotion Detection
├── emotion_model.h5           # Trained Model
├── fer2013.csv               # Dataset
├── README.md
└── requirements.txt
```

---

## 🧠 Dataset

**Dataset:** FER2013

* Total Images: **35,887**
* Image Size: **48 × 48**
* Grayscale Images
* 7 Emotion Classes

### Emotion Classes

| Label | Emotion  |
| ----- | -------- |
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

---

## 🏗 CNN Architecture

* Conv2D (32 Filters)

* Batch Normalization

* Conv2D (32 Filters)

* MaxPooling

* Dropout

* Conv2D (64 Filters)

* Batch Normalization

* Conv2D (64 Filters)

* MaxPooling

* Dropout

* Conv2D (128 Filters)

* Batch Normalization

* Conv2D (128 Filters)

* MaxPooling

* Dropout

* Flatten Layer

* Dense (512)

* Dropout

* Output Layer (Softmax)

---

## ⚙ Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📊 Model Performance

| Metric          | Value                    |
| --------------- | ------------------------ |
| Test Accuracy   | **63.97%**               |
| Training Method | CNN                      |
| Optimizer       | Adam                     |
| Loss Function   | Categorical Crossentropy |
| Epochs          | 41 (Early Stopping)      |

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
jupyter notebook emotion.ipynb
```

### Run Real-Time Emotion Detection

```bash
python apps.py
```

Press **Q** to close the webcam.

---

## 📈 Evaluation

The model performance is evaluated using:

* Test Accuracy
* Classification Report
* Confusion Matrix
* Accuracy Graph
* Loss Graph

---

## 📷 Real-Time Detection

The webcam application:

* Detects faces using OpenCV Haar Cascade.
* Preprocesses the detected face.
* Predicts one of seven emotions.
* Displays the predicted emotion and confidence score in real time.

---

## 🔮 Future Improvements

* Improve performance using Transfer Learning (MobileNetV2, EfficientNet, ResNet50)
* Improve low-light detection
* Deploy using Streamlit or Flask
* Convert model to TensorFlow Lite
* Add video file emotion detection

---

## 📚 References

1. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

2. Ian Goodfellow et al. "Challenges in Representation Learning: A Report on Three Machine Learning Contests." arXiv:1307.0414.

3. FER2013 Dataset:
   https://www.kaggle.com/datasets/msambare/fer2013

4. TensorFlow Documentation:
   https://www.tensorflow.org/

5. OpenCV Documentation:
   https://docs.opencv.org/

6. Keras Documentation:
   https://keras.io/

---



---
