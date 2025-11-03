# Traffic Sign Classification Project

This project implements a deep learning model to classify traffic signs using a Convolutional Neural Network (CNN). The model is built with TensorFlow/Keras and trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which includes 43 different classes of traffic signs.

## 📋 Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## 🎯 Introduction

Traffic sign classification is a critical component for autonomous driving systems. It allows a vehicle to recognize and interpret road signs to make safe driving decisions. This project leverages deep learning to build a high-accuracy classifier for this purpose.

## 📊 Dataset

The model is trained and tested on the **German Traffic Sign Recognition Benchmark (GTSRB)**. This dataset contains over 50,000 images across 43 classes.

- **Official Dataset Link:** [GTSRB Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
- **Kaggle Version:** [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## 💻 Technologies Used

- Python
- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/s0maye88/traffic-sign-classification.git
    cd traffic-sign-classification
    ```

2.  **Install dependencies:**
    It's recommended to create a `requirements.txt` file, but you can also install the libraries manually:
    ```bash
    pip install numpy pandas tensorflow scikit-learn matplotlib jupyterlab
    ```

3.  **Run the notebooks:**
    Open the Jupyter Notebooks to see the process step-by-step. The main notebook contains all the logic for training, evaluation, and visualization.
    -   **`traffic_sign_image_classification_CNN_improved.ipynb`**: This is the main notebook. Open it to see data loading, model training, and performance evaluation.
    -   **`demo.ipynb`**: This notebook can be used to test the trained model on single images.

## 📂 Project Structure

The repository is organized as follows:

```
traffic-sign-classification/
├── traffic_sign_image_classification_CNN_improved.ipynb  # Main notebook for training & evaluation
├── demo.ipynb                                          # Notebook for testing predictions
├── Meta/                                               # Folder with sample images for each class
├── .gitignore
└── README.md
```

## 🏗️ Model Architecture

The model is a **Convolutional Neural Network (CNN)** with the following key layers:
- `Conv2D` layers for feature extraction.
- `MaxPooling2D` layers to reduce dimensionality.
- `Dropout` layers to prevent overfitting.
- `Flatten` layer to prepare data for the final classification layers.
- `Dense` layers for the final classification.

The model is compiled with the `Adam` optimizer and `sparse_categorical_crossentropy` loss function.

## 📈 Results

The model achieves an accuracy of approximately **98%** on the test set.

## 📜 License

This project is licensed under the MIT License. Copyright (c) 2024, s0maye88.
