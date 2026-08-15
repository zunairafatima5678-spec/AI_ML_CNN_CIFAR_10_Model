# CNN CIFAR-10 Image Classifier

## Overview
This project provides a Streamlit web application that utilizes a trained Convolutional Neural Network (CNN) to classify images into one of the 10 categories from the CIFAR-10 dataset.

## Features
- **Image Upload**: Easily upload JPG, JPEG, or PNG images.
- **CNN-based Classification**: Leverages a deep learning model for accurate image categorization.
- **CIFAR-10 Dataset**: Specialized for the 10 common object classes in CIFAR-10.
- **Confidence Score**: Displays the model's confidence in its top prediction.
- **Top-3 Predictions**: Shows the top three predicted classes with their respective probabilities.
- **Streamlit Interface**: A professional, user-friendly web interface.
- **GitHub Deployment Ready**: Configured for seamless deployment via GitHub to platforms like Streamlit Community Cloud.

## Model
The CNN architecture used in this application follows a common pattern for image classification:

```
Input (32x32x3)
↓
Convolution (32 filters, 3x3, ReLU)
↓
Max Pooling (2x2)
↓
Dropout (0.25)
↓
Convolution (64 filters, 3x3, ReLU)
↓
Max Pooling (2x2)
↓
Dropout (0.25)
↓
Flatten
↓
Dense (128 units, ReLU)
↓
Dropout (0.5)
↓
Dense (10 units, Softmax)
```

The model was trained on the CIFAR-10 dataset, processing 32x32 pixel RGB images and normalizing pixel values to the [0, 1] range.

## CIFAR-10 Classes
The 10 classes the model can classify are:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Project Structure
```text
Deployment/
│
├── app.py              # The main Streamlit application script
├── requirements.txt    # List of Python dependencies
├── README.md           # Project description and instructions
├── .gitignore          # Files/directories to ignore in Git
│
├── model/
│   └── cnn_cifar10.keras # The trained Keras model file
│
└── assets/             # (Optional) Directory for additional assets like images, CSS, etc.
```

## Installation
1.  **Clone the repository** (if hosted on GitHub) or navigate to the `Deployment` folder.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Run Locally
Once the dependencies are installed, you can run the Streamlit application from the `Deployment` directory:

```bash
streamlit run app.py
```

This will open the application in your web browser.

## Deployment
To deploy this application using Streamlit Community Cloud (or similar platforms):

1.  **Push your `Deployment` folder to a GitHub repository.** Ensure the `model/cnn_cifar10.keras` file is included.
2.  **Go to the Streamlit Community Cloud website** and sign in.
3.  **Create a new app** and connect it to your GitHub repository.
4.  **Specify the path to your `app.py` file** (e.g., `Deployment/app.py` if your repository root contains the `Deployment` folder).
5.  **Set the Python version** (e.g., Python 3.9 or higher, compatible with your `requirements.txt`).
6.  **Deploy the app!** Streamlit will automatically install dependencies from `requirements.txt` and launch your application.
