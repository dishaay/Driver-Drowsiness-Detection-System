# Driver Drowsiness Detection System

A real-time driver drowsiness detection system developed using **Python, Convolutional Neural Networks (CNN), OpenCV, and CustomTkinter**. The application monitors a driver's eye state through a webcam and triggers an audible alert when prolonged eye closure is detected.

## Overview

Driver fatigue is a major road safety concern, as it reduces concentration, alertness, and reaction time. This project aims to detect potential drowsiness by analyzing eye conditions using computer vision and deep learning.

The system uses OpenCV for face and eye detection, a custom-trained CNN model for eye-state classification, and a graphical user interface for real-time monitoring.

## Features

- Real-time webcam monitoring.
- Face and eye detection using OpenCV Haar Cascades.
- Eye-state classification using a custom CNN model.
- Detection of prolonged eye closure.
- Audible warning when potential drowsiness is detected.
- Live video feed with detection indicators.
- Interactive desktop interface using CustomTkinter.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| TensorFlow / Keras | CNN model training and prediction |
| OpenCV | Webcam processing and eye detection |
| NumPy | Numerical operations |
| CustomTkinter | Graphical user interface |
| Pillow (PIL) | Image processing and GUI integration |
| Winsound | Audible drowsiness alert |
| Jupyter Notebook | Model experimentation |

## Project Structure

```text
Drowsiness-Driver-Detection/
│
├── app.py
├── train_model.py
├── drowsiness_detection.ipynb
├── drowsiness_cnn.keras
├── archive/
├── .gitignore
└── README.md
```

**File descriptions:**

- `app.py` – Main application with webcam monitoring, CNN predictions, GUI, and alarm functionality.
- `train_model.py` – Script used to train the CNN model.
- `drowsiness_detection.ipynb` – Notebook for experimentation and model development.
- `drowsiness_cnn.keras` – Saved trained CNN model.
- `archive/` – Dataset containing eye images used during training.

## Working Methodology

**1. Webcam Capture**

The system captures live video frames using OpenCV.

**2. Face and Eye Detection**

Haar Cascade classifiers identify the face and eye regions in each frame.

**3. Image Preprocessing**

Detected eye images are resized and normalized before being passed to the CNN model.

**4. CNN Classification**

The trained CNN classifies the detected eye images into two categories:

- Open Eyes
- Closed Eyes

**5. Drowsiness Detection**

The application monitors eye-closure predictions over time to distinguish prolonged closure from normal blinking.

**6. Alert Generation**

When prolonged eye closure exceeds the configured threshold, the system activates an audible alarm to alert the driver.

## Model Training

The project uses a custom Convolutional Neural Network trained on images of open and closed eyes.

The CNN learns visual features associated with different eye states through convolutional layers and classifies them using the final prediction layer.

During development, the model achieved approximately **98% validation accuracy** on the eye-state classification dataset.

This accuracy represents classification performance on the validation dataset and does not guarantee equivalent performance under real driving conditions.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Drowsiness-Driver-Detection.git
cd Drowsiness-Driver-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install tensorflow opencv-python numpy customtkinter pillow
```

### 4. Add the trained model

Ensure that `drowsiness_cnn.keras` is present in the project directory.

If the trained model is not included in the repository, obtain it separately or regenerate it using the training script and required dataset.

### 5. Run the application

```bash
python app.py
```

Allow webcam access when prompted.

**Note:** The current application uses Windows-specific audio functionality through `winsound`.

## Applications

- Driver fatigue awareness and monitoring.
- Computer vision and deep learning demonstrations.
- Research and educational projects related to road safety.
- Prototype development for driver monitoring systems.

## Limitations

- Detection accuracy may decrease in poor lighting.
- Glasses and head movements may interfere with eye detection.
- Haar Cascade detection may fail at certain camera angles.
- The system relies primarily on eye closure rather than multiple fatigue indicators.
- Performance has not been validated for real-world driving safety.

## Future Enhancements

- Integration of facial landmarks for improved eye tracking.
- Head-pose estimation and yawning detection.
- Improved performance under different lighting conditions.
- Support for additional operating systems.
- Optimization for embedded and edge devices.

## Conclusion

The Driver Drowsiness Detection System demonstrates how deep learning and computer vision can be combined to identify prolonged eye closure in real time. By integrating CNN-based eye classification, OpenCV detection, and an audible warning mechanism, the project provides a practical prototype for driver fatigue monitoring.

**Disclaimer:** This project is intended for educational and experimental purposes and should not be relied upon as a certified driver safety system.
