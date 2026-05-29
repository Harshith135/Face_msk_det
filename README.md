# Face Mask Detection

## Project Overview

This project detects whether a person is wearing a face mask or not using a Convolutional Neural Network (CNN) model built with TensorFlow and deployed using Streamlit.

## Features

- Detects Face Mask and No Face Mask classes
- User-friendly Streamlit web interface
- Deep Learning model trained using TensorFlow/Keras
- Image-based prediction

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow (PIL)

## Project Files

- `app1.py` – Streamlit application
- `train_model.py` – Model training script
- `face_mask_cnn_model.keras` – Trained CNN model
- `requirements.txt` – Project dependencies

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app1.py
```

3. Open the local URL shown in the terminal and upload an image for prediction.

## Output

The application predicts whether the uploaded image contains a person wearing a face mask or not.

## Author

Harshith H
