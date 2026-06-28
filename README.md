# ⛽ Global Petrol Price Prediction using Machine Learning

## 📌 Project Overview

This project predicts petrol prices using machine learning techniques based on economic and geographical factors such as region, income level, subsidy level, Brent crude oil prices, and tax percentages.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis, model building, evaluation, and model serialization using Scikit-Learn pipelines.

---

## 🎯 Problem Statement

Petrol prices vary significantly across countries due to several factors such as fuel taxes, government subsidies, crude oil prices, and economic conditions.

The objective of this project is to analyze these factors and build a machine learning model capable of predicting petrol prices.

---

## 📊 Dataset Features

| Feature         | Description                      |
| --------------- | -------------------------------- |
| region          | Geographic region                |
| income_level    | Economic category of the country |
| subsidy_level   | Government fuel subsidy level    |
| brent_crude_usd | Brent crude oil price            |
| tax_percentage  | Fuel tax percentage              |

### Target Variable

* **petrol_usd_liter**

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

* Distribution analysis
* Count plots
* Correlation heatmap
* Feature relationship analysis
* Outlier analysis
* Residual analysis
* Data visualization

---

## ⚙️ Data Preprocessing

### One-Hot Encoding

Applied to:

* region

### Ordinal Encoding

Applied to:

* income_level
* subsidy_level

### Feature Scaling

* StandardScaler

---

## 🤖 Machine Learning Models Implemented

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet Regression

---

## 📈 Model Evaluation Metrics

Models were evaluated using:

* R² Score
* Adjusted R² Score
* Cross Validation
* Residual Analysis

---

## 🔄 Log Transformation

The target variable was log-transformed to improve homoscedasticity and reduce residual variance, leading to improved regression performance.

Predictions are converted back to the original scale using:

```python
np.exp(prediction)
```

---

## 🏆 Final Model

The final model uses:

* ColumnTransformer
* Pipeline
* Ridge Regression

The complete preprocessing pipeline and trained model are saved as:

```text
petrol_price_model.pkl
```

---

## 📂 Project Structure

```text
Global-Petrol-Price-Prediction/
│
├── Petrol_Price_Prediction_LinearRegression.ipynb
├── petrol_price_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone git clone https://github.com/Monika1870/Global-Petrol-Price-Prediction.git

cd Global-Petrol-Price-Prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 💾 Load the Model

```python
import joblib

model = joblib.load("petrol_price_model.pkl")
```

---

## 🔮 Example Prediction

```python
import pandas as pd
import numpy as np
import joblib

model = joblib.load("petrol_price_model.pkl")

new_data = pd.DataFrame({
    "region": ["Asia"],
    "income_level": ["Middle"],
    "subsidy_level": ["Medium"],
    "brent_crude_usd": [85],
    "tax_percentage": [20]
})

prediction_log = model.predict(new_data)

prediction = np.exp(prediction_log)

print("Predicted Petrol Price:", round(prediction[0], 2))
```

---

## 🛠️ Libraries Used

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-Learn
* Joblib

---

## 👩‍💻 Author

**Monika Gautam**

* LinkedIn: https://www.linkedin.com/in/monika-gautam-7b09b0172/
* GitHub: https://github.com/Monika1870

