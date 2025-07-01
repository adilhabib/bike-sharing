# Bike Rental Forecasting Project
# Author: Adil Habib
# Goal: Forecast bike rentals using weather, time, and calendar features

# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# ---
# 2. LOAD DATA
hour = pd.read_csv('train.csv', parse_dates=['datetime'])

# ---
# 3. FEATURE ENGINEERING
hour['hour'] = hour['datetime'].dt.hour
hour['weekday'] = hour['datetime'].dt.weekday
hour['month'] = hour['datetime'].dt.month
hour['year'] = hour['datetime'].dt.year.map({2011: 0, 2012: 1})
hour['is_weekend'] = hour['weekday'].isin([5, 6]).astype(int)

# Simplify weather categories
hour['weather_simple'] = hour['weather'].replace({
    1: 'clear',
    2: 'mist',
    3: 'light_precip',
    4: 'heavy_precip'
})


hour.rename(columns={'weather': 'weathersit'}, inplace=True)


# ---
# 4. EXPLORATORY DATA ANALYSIS
plt.figure(figsize=(12, 6))
sns.boxplot(x='hour', y='count', data=hour)
plt.title('Hourly Bike Rentals')
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='weather_simple', y='count', data=hour)
plt.title('Rentals by Weather Type')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='is_weekend', y='count', data=hour)
plt.title('Weekday vs Weekend Rentals')
plt.xticks([0, 1], ['Weekday', 'Weekend'])
plt.show()

# ---
# 5. DATA PREPARATION
X = hour.drop(['datetime', 'casual', 'registered', 'count'], axis=1)
y = hour['count']

# One-hot encoding
X = pd.get_dummies(X, columns=['season', 'weather_simple', 'month', 'weekday', 'hour'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---
# 6. LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
print("Linear Regression RMSE:", round(rmse_lr, 2))

# ---
# 7. RANDOM FOREST MODEL
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest RMSE:", round(rmse_rf, 2))

# ---
# 8. FEATURE IMPORTANCE
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.title('Top 10 Feature Importances')
plt.xlabel('Importance')
plt.show()

# ---
# 9. SIMULATED SCENARIO PREDICTION
sample = X_test.iloc[0].copy()
sample.loc[:] = 0
sample['hour_17'] = 1
sample['season_2'] = 1
sample['atemp'] = 28
sample['humidity'] = 85
sample['windspeed'] = 25
sample['weather_simple_heavy_precip'] = 1
sample['workingday'] = 1
sample['is_weekend'] = 0
sample['year'] = 1

prediction = rf.predict(pd.DataFrame([sample]))
print("Predicted Rentals on Rainy Weekday at 5PM:", int(prediction[0]))
