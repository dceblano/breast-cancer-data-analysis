# Breast Cancer Prediction App

This project implements a machine learning model to predict breast cancer based on the Breast Cancer Wisconsin (Diagnostic) dataset. The app is built using Streamlit for the user interface, allowing users to interact with the model and see predictions in real-time.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [App Interface](#app-interface)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Breast Cancer Prediction App allows users to adjust feature values through sliders and see the model's prediction of whether the breast cancer is malignant or benign. The model used is an Artificial Neural Network (ANN) implemented with scikit-learn.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

## Model Training
The model training process involves:
1. Loading and preprocessing the dataset.
2. Performing feature selection using `SelectKBest`.
3. Training an ANN model using `MLPClassifier` from scikit-learn.
4. Saving the trained model and preprocessing objects using pickle.

## App Interface
The app interface is built using Streamlit, allowing users to:
- Adjust the values of features using sliders.
- Make predictions based on the adjusted values.
- View the prediction result and the probability of malignancy and benignancy.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation
1. **Clone the repository**:
    ```sh
    git clone https://github.com/dceblano/breast-cancer-data-analysis.git
    cd breast-cancer-prediction-app
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

### Usage
1. Open the web browser and go to the address provided by Streamlit (usually `http://localhost:8501`).
2. Adjust the values of the features using the sliders.
3. Click the "Predict" button to see the model's prediction and the probability of malignancy or benignancy.

## Contributing
Contributions are welcome! Please fork this repository and submit a pull request for any changes you would like to make.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

