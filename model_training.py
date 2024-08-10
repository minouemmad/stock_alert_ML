import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

def engineer_features(df):
    # Print the column names to debug
    print("Original columns:", df.columns)

    # Check if 'transaction_code' column is in the DataFrame
    if 'transaction_code' not in df.columns:
        print("'transaction_code' column is missing from the DataFrame.")
        return df  # Exit the function if the required column is missing

    # Rename columns for consistency
    df.rename(columns={'trade_date': 'date', 'transaction_codes': 'transaction_code', 'total_value_of_trade': 'total_value'}, inplace=True)
    
    # Print the column names after renaming to debug
    print("Renamed columns:", df.columns)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create transaction type feature
    df['transaction_type'] = df['transaction_code'].apply(lambda x: 1 if x == 'P' else -1 if x == 'S' else 0)
    
    # Log transform of price per share
    df['log_price'] = df['price_per_share'].apply(lambda x: np.log(x + 1))
    
    # Value in thousands
    df['value_in_thousands'] = df['total_value'] / 1000
    
    # Extract hour of day from date
    df['hour_of_day'] = df['date'].dt.hour
    
    # Create lag features for predicting future prices
    df['price_lag_1'] = df['log_price'].shift(1)
    df['price_lag_2'] = df['log_price'].shift(2)
    
    # Drop rows with NaNs resulting from lag features
    df.dropna(inplace=True)
    
    # Select relevant features
    relevant_columns = [
        'date', 'ticker', 'transaction_type', 'log_price', 'value_in_thousands', 
        'num_shares', 'hour_of_day', 'price_lag_1', 'price_lag_2'
    ]
    
    df = df[relevant_columns]
    return df

def train_model(df):
    X = df.drop(columns=['date', 'log_price'])
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    model = XGBRegressor()
    randomized_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, cv=3, scoring='neg_mean_squared_error', verbose=2, random_state=42, n_jobs=-1)
    randomized_search.fit(X_train, y_train)

    best_model = randomized_search.best_estimator_

    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return best_model

if __name__ == "__main__":
    # Ensure that the features.pkl file exists in the directory
    data = pd.read_pickle('features.pkl')
    print("Loaded data columns:", data.columns)  # Print loaded data columns
    feature_data = engineer_features(data)
    if 'transaction_type' in feature_data.columns:  # Proceed only if features are correctly engineered
        model = train_model(feature_data)
        joblib.dump(model, 'stock_price_model.pkl')
    else:
        print("Feature engineering failed, model training skipped.")
