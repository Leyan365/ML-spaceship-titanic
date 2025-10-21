# 🚀 Spaceship Titanic — Machine Learning Solution

A data science project built for the **Kaggle Spaceship Titanic** competition.

This notebook explores feature engineering, model experimentation, and ensemble blending to predict whether passengers were transported during an interstellar collision.

---

## 📘 Overview

The Spaceship Titanic dataset presents a binary classification challenge similar to the classic Titanic problem — but set in space!

Each passenger’s information (age, spending habits, home planet, destination, and more) is used to predict the **Transported** target variable.

---

## ⚙️ Project Workflow

### 1. Data Preprocessing & Feature Engineering
- Handled missing values with median imputation for numeric features.
- **Created new features:**
  - `TotalSpending`: Sum of all onboard expenses.
  - `Log_TotalSpending`: Log-transformed spending for normalization.
  - `GroupSize`: Number of passengers sharing the same group ID.
- Extracted `Deck`, `Num`, and `Side` from the `Cabin` field.
- One-hot encoded categorical variables (`HomePlanet`, `Destination`, `Deck`, `Side`).

---

### 2. Models Trained
The following models were trained and cross-validated:
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  
- LightGBM Classifier  
- CatBoost Classifier (tuned with cross-validation)

---

### 3. Model Blending
The top three models (**XGBoost**, **LightGBM**, and **CatBoost**) were blended to form a weighted ensemble.  
Weights were optimized using validation AUC scores, resulting in the **best overall performance** among all tested methods.

---

## 🧩 Submission Results
- **Kaggle Public Leaderboard Score:** `0.74444`  
- **Model:** Blended XGB + LGB + CatBoost  
- **Submission Command:**  
  ```bash
  kaggle competitions submit -c spaceship-titanic -f submission_blended.csv -m "Blended XGB+LGB+CatBoost model"
