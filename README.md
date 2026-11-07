# 🤖 Customer Churn Prediction ML App

A lightweight, production-ready machine learning web application built with Python and Streamlit to predict customer churn risk and retention based on user behavior patterns.

---

## 🚀 Key Features
* **Custom Machine Learning Model:** Trained classifier model designed to evaluate key customer behavioral metrics (Age, Monthly Spend, Support Calls, and Tenure).
* **Interactive Web Interface:** Built with **Streamlit** allowing users to adjust sliders and input parameters in real-time for instant predictions.
* **Risk Assessment & Recommendations:** Automatically flags high-risk vs. low-risk customers with confidence scores and provides actionable business recommendations.
* **Pure Python Architecture:** Optimized to run smoothly and independently without complex binary dependencies.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Machine Learning:** Custom Classifier & Pickle (`churn_model.pkl`)
* **Web Framework / UI:** Streamlit
* **Data Processing:** Pure Python Logic

---

## 📂 Project Structure
```text
Python_ML_Prediction_App/
│
├── train_model.py      # ML Model Training Script
├── churn_model.pkl     # Saved Machine Learning Model Artifact
├── app.py              # Streamlit Web Application
└── README.md           # Project Documentation
