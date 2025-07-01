# 🚲 Bike Rental Forecasting with Machine Learning

## 📌 Overview

This project uses the Kaggle **Bike Sharing Demand** dataset to build a machine learning model that predicts hourly bike rental counts in Washington, D.C. The goal is to help a bike-sharing company optimize fleet allocation and operational planning.

---

## 📊 Dataset

**Source:** [Kaggle Competition - Bike Sharing Demand](https://www.kaggle.com/competitions/bike-sharing-demand)

The dataset includes:

* Hourly timestamps (datetime)
* Weather conditions (temperature, humidity, windspeed, etc.)
* Temporal features (hour, day of week, month, year)
* Rental counts (`casual`, `registered`, `count`)

---

## 🔍 Key Features Engineered

* **hour**: Hour of the day
* **is\_weekend**: Weekend flag
* **weather\_simple**: Simplified weather condition
* **season, month, weekday**: One-hot encoded categorical features

---

## 🚀 Modeling Approach

### 1. **Linear Regression** (baseline)

* RMSE: \~100.14

### 2. **Random Forest Regressor** (improved)

* RMSE: \~47.93
* Captured nonlinear trends like commute peaks

### 🔍 Top Influential Features

* `atemp`: Feels-like temperature
* `hour_17`, `hour_18`: Evening commute hours
* `year`: Overall usage growth

---

## 🧪 Business Scenario Simulation

> *“On a hot weekday at 5 PM with heavy rain, expected demand drops to \~X bikes.”*

This demonstrates a model application for **demand forecasting and bike repositioning**.

---

## 🛠 Tech Stack

* Python (Pandas, scikit-learn, Seaborn, Matplotlib)
* Jupyter Notebook
* GitHub for version control

---

## 📂 File Structure

```
├── train.csv              # Raw data
├── bike_forecasting.ipynb # Complete code & analysis
├── README.md              # Project overview
```

---

## 📈 Results & Insights

* Rentals peak during commute hours
* Rain and cold weather reduce demand
* Registered users dominate weekday usage

---

## 📣 About the Author

**Adil Habib** – Aspiring data analyst with strong analytical and storytelling skills. Seeking roles in data-driven decision making, forecasting, and BI.

---

## 📬 Contact

* LinkedIn: \adil-habib
* GitHub: \adilhabib
