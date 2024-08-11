# model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_features():
    try:
        # Load engineered features from the Parquet file
        features = pd.read_parquet('/data/processed/features.parquet', engine='pyarrow')
        return features
    except FileNotFoundError:
        print("Feature file not found. Ensure that feature_engineering.py has been run successfully.")
        return None

def train_model(features):
    # Drop or convert datetime columns
    if 'date' in features.columns:
        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day'] = features['date'].dt.day
        features['day_of_week'] = features['date'].dt.dayofweek
        features = features.drop(columns=['date'])  # Drop the original date column

    # Convert categorical features to numeric using one-hot encoding or similar methods
    features = pd.get_dummies(features, drop_first=True)

    # Replace infinite values with NaN
    features.replace([float('inf'), float('-inf')], float('nan'), inplace=True)

    # Handle NaN values
    features.fillna(features.mean(), inplace=True)

    # Define the target variable (price)
    X = features.drop(columns=['log_price'])
    y = features['log_price']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model - using RandomForestRegressor for example
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    print(f"Model training completed. RMSE: {rmse}")

    return model



def save_model(model):
    # Save the trained model to a file using joblib
    joblib.dump(model, '/models/stock_prediction_model.pkl')
    print("Model saved to stock_prediction_model.pkl.")

if __name__ == "__main__":
    # Load features
    features = load_features()

    if features is not None:
        # Train the model
        model = train_model(features)

        # Save the trained model
        save_model(model)
