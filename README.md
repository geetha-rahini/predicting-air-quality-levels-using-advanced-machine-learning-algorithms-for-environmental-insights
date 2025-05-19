# Predicting Air Quality Levels Using Advanced Machine Learning Algorithms for Environmental Insights

# This script fetches air quality data from OpenAQ API, preprocesses it,
# performs exploratory data analysis, trains an XGBoost model to predict PM2.5 levels,
# and visualizes results.

# Required Libraries:
# pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, requests

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")

def fetch_openaq_data(city="Los Angeles", parameter="pm25", limit=5000):
    """Fetch pm25 air quality data from OpenAQ API for a specified city."""
    base_url = "https://api.openaq.org/v2/measurements"
    all_results = []
    page = 1
    print(f"Fetching data for city: {city}, parameter: {parameter}...")
    while len(all_results) < limit:
        params = {
            "city": city,
            "parameter": parameter,
            "limit": 1000,  # per request max
            "page": page,
            "order_by": "datetime",
            "sort": "asc"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data, status code: {response.status_code}")
            break
        data = response.json()
        results = data.get('results', [])
        if not results:
            break
        all_results.extend(results)
        if len(results) < 1000:
            # No more pages
            break
        page += 1
    print(f"Fetched {len(all_results)} records.")
    return all_results[:limit]

def load_and_process(data):
    """Load JSON data into DataFrame and preprocess."""
    df = pd.json_normalize(data)
    df = df[['date.utc', 'value', 'unit', 'location', 'city', 'country']]
    df.rename(columns={'date.utc': 'datetime', 'value': 'pm25'}, inplace=True)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def feature_engineering(df):
    """Create datetime features for modeling."""
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['pm25_rolling3'] = df['pm25'].rolling(window=3, min_periods=1).mean()
    return df

def plot_time_series(df, city):
    plt.figure(figsize=(15,5))
    sns.lineplot(x='datetime', y='pm25', data=df)
    plt.title(f"PM2.5 Levels Over Time in {city}")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.show()

def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    print("Training XGBoost model...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R2 Score: {r2:.3f}")
    return y_pred

def plot_predictions(df, y_train_len, y_test, y_pred):
    plt.figure(figsize=(15,5))
    plt.plot(df['datetime'].iloc[y_train_len:y_train_len+len(y_test)], y_test, label='Actual')
    plt.plot(df['datetime'].iloc[y_train_len:y_train_len+len(y_test)], y_pred, label='Predicted')
    plt.title("Actual vs Predicted PM2.5 Levels")
    plt.xlabel("Date")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.legend()
    plt.show()
    
def plot_feature_importance(model, features):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    sns.barplot(x=[features[i] for i in indices], y=importances[indices], palette='viridis')
    plt.title('Feature Importance')
    plt.show()

def main():
    city_name = "Los Angeles"
    data = fetch_openaq_data(city=city_name, parameter="pm25", limit=5000)
    df = load_and_process(data)
    
    print(f"Data preview:\n{df.head()}")
    print(f"Data shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    plot_time_series(df, city_name)
    
    df = feature_engineering(df)
    
    features = ['hour', 'day', 'month', 'dayofweek', 'pm25_rolling3']
    target = 'pm25'
    X = df[features]
    y = df[target]
    
    # Train-test split preserving time order
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    
    plot_predictions(df, len(y_train), y_test, y_pred)
    plot_feature_importance(model, features)

if __name__ == "__main__":
    main()

