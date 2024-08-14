import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def engineer_features(df):
    df['price_per_share'] = df['price_per_share'].replace(0, np.nan)
    df['log_price'] = df['price_per_share'].apply(lambda x: np.log(x) if pd.notnull(x) else np.nan)

    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df.drop(columns=['date'], inplace=True)

    df.dropna(subset=['log_price'], inplace=True)
    return df

def predict_new_data(new_data):
    model = joblib.load('models\\stock_prediction_model.pkl')

    feature_data = engineer_features(new_data)

    X = pd.get_dummies(feature_data.drop(columns=['price_per_share', 'log_price']), drop_first=True)
    model_features = model.feature_names_in_

    missing_cols = set(model_features) - set(X.columns)
    for col in missing_cols:
        X[col] = 0

    X = X[model_features]

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)

    predictions = model.predict(X)

    return predictions

def plot_predictions_vs_actual(predictions, actual_prices):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_prices, predictions, alpha=0.5)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Predicted vs Actual Prices')

    # Option 1: Use log scale for better clarity
    plt.xscale('log')
    plt.yscale('log')

    # Diagonal reference line for perfect predictions
    plt.plot([min(actual_prices), max(actual_prices)], [min(actual_prices), max(actual_prices)], 'r--')

    plt.show()

def plot_residuals(predictions, actual_prices):
    residuals = predictions - actual_prices
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_prices, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Prices')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title('Residuals vs Actual Prices')
    plt.show()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_error_metrics(predictions, actual_prices):
    mae = mean_absolute_error(actual_prices, predictions)
    mse = mean_squared_error(actual_prices, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predictions)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R²): {r2}")


import seaborn as sns

def plot_error_distribution(predictions, actual_prices):
    residuals = predictions - actual_prices
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals (Predicted - Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.show()


if __name__ == "__main__":
    new_data = pd.read_pickle('data\\raw\\insider_trading_test.pkl')

    predictions = predict_new_data(new_data)

    predicted_prices = np.exp(predictions)

    actual_prices = new_data['price_per_share'].values

    # Error Metrics
    calculate_error_metrics(predicted_prices, actual_prices)

    # Residuals Plot
    plot_residuals(predicted_prices, actual_prices)

    # Error Distribution Plot
    plot_error_distribution(predicted_prices, actual_prices)

