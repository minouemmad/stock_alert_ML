import pandas as pd
import numpy as np

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
        'num_shares', 'hour_of_day', 'price_lag_1', 'price_lag_2'
    ]
    
    df = df[relevant_columns]
    return df

if __name__ == "__main__":
    # Read data from Pickle file
    data = pd.read_pickle('insider_trading_data.pkl')
    
    # Engineer features
    feature_data = engineer_features(data)
    
    # Save engineered features in various formats
    feature_data.to_csv('features.csv', index=False)
    feature_data.to_pickle('features.pkl')
    feature_data.to_hdf('features.h5', key='df', mode='w')
    feature_data.to_parquet('features.parquet')
    feature_data.to_feather('features.feather')
