#feature_engineering.py
import pandas as pd
import numpy as np
import os

def engineer_features(df):
    # Rename columns for consistency
    df.rename(columns={
        'trade_date': 'date', 
        'transaction_codes': 'transaction_code', 
        'total_value_of_trade': 'total_value'
    }, inplace=True)
    
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
        'num_shares', 'hour_of_day', 'price_lag_1', 'price_lag_2',
        'owner_type', 'percent_change_in_shares', 'division', 'major_group'
    ]
    
    df = df[relevant_columns]
    return df

if __name__ == "__main__":
    # Check if the environment variables are set
    if not os.path.exists('insider_trading_data.pkl'):
        print("Pickle file not found. Ensure that data_extraction.py has been run successfully.")
    else:
        try:
            # Read data from Pickle file
            data = pd.read_pickle('insider_trading_data.pkl')
            
            # Engineer features
            feature_data = engineer_features(data)
            
            feature_data.to_parquet('features.parquet', engine='pyarrow')
            # Save engineered features in various formats except HDF5
            feature_data.to_csv('/data/processed/features.csv', index=False)
            feature_data.to_pickle('/data/processed/features.pkl')
            feature_data.to_parquet('/data/processed/features.parquet')
            feature_data.to_feather('/data/processed/features.feather')
            
            print("Feature engineering completed and data saved in multiple formats.")
        
        except ModuleNotFoundError as e:
            print(f"Error: {e}. Make sure all required modules are installed.")
