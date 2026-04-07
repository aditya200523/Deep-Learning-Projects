# Customer Churn Prediction using Artificial Neural Network

A deep learning project that predicts whether a bank customer is likely to churn, built with TensorFlow/Keras and deployed via a Streamlit web application.

---

## Overview

Customer churn prediction helps banks proactively identify customers at risk of leaving. This project trains an ANN on the [Churn Modelling dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) and provides an interactive web interface for real-time predictions.

---

## Project Structure

```
├── app.py                        # Streamlit web application
├── Churn_Modelling.csv           # Dataset
├── model.h5                      # Trained ANN model
├── scaler.pkl                    # StandardScaler
├── label_encoder_gender.pkl      # LabelEncoder for Gender
├── onehot_encoder_geography.pkl  # OneHotEncoder for Geography
└── logs/                         # TensorBoard training logs
```

---

## Model Architecture

| Layer  | Units | Activation |
|--------|-------|------------|
| Input  | 12    | —          |
| Dense  | 64    | ReLU       |
| Dense  | 32    | ReLU       |
| Output | 1     | Sigmoid    |

- **Optimizer:** Adam  
- **Loss:** Binary Crossentropy  
- **Callbacks:** EarlyStopping (patience=10), TensorBoard

---

## Features Used

| Feature           | Type        | Encoding            |
|-------------------|-------------|---------------------|
| CreditScore       | Numerical   | StandardScaler      |
| Geography         | Categorical | OneHotEncoder       |
| Gender            | Categorical | LabelEncoder        |
| Age               | Numerical   | StandardScaler      |
| Tenure            | Numerical   | StandardScaler      |
| Balance           | Numerical   | StandardScaler      |
| NumOfProducts     | Numerical   | StandardScaler      |
| HasCrCard         | Binary      | —                   |
| IsActiveMember    | Binary      | —                   |
| EstimatedSalary   | Numerical   | StandardScaler      |

---

## Getting Started

### Prerequisites

```bash
pip install pandas scikit-learn tensorflow streamlit
```

### Running the Notebook

1. Clone the repository
2. Place `Churn_Modelling.csv` in the root directory
3. Run the notebook cells in order — preprocessing → training → prediction
4. Trained artifacts (`model.h5`, `.pkl` files) will be saved automatically

### Running the Streamlit App

```bash
streamlit run app.py
```

---

## Training Monitoring

TensorBoard logs are saved under `logs/fit/`. To visualize training:

```bash
tensorboard --logdir logs/fit
```

---

## Sample Prediction

```python
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
# Output: "The customer is not likely to churn"
```

---

## Tech Stack

- **Python** — Core language
- **TensorFlow / Keras** — ANN model
- **Scikit-learn** — Preprocessing
- **Pandas** — Data manipulation
- **Streamlit** — Web application
- **TensorBoard** — Training visualization
- **Pickle** — Model serialization

---

## Dataset

[Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) — 10,000 bank customer records with 14 features.

---
