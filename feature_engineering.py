#Create features from the raw data to be used for training the model.

import pandas as pd

def engineer_features(df):
    df['date'] = pd.to_datetime(df['date'])
    df['transaction_type'] = df['transaction_code'].apply(lambda x: 1 if x == 'P' else -1 if x == 'S' else 0)
    df['log_price'] = df['price_per_share'].apply(lambda x: np.log(x + 1))
    df['value_in_thousands'] = df['total_value'] / 1000
    df['hour_of_day'] = df['date'].dt.hour

    # Create lag features for predicting future prices
    df['price_lag_1'] = df['log_price'].shift(1)
    df['price_lag_2'] = df['log_price'].shift(2)

    # Drop rows with NaNs resulting from lag features
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = pd.read_csv('insider_trading_data.csv')
    feature_data = engineer_features(data)
    feature_data.to_csv('features.csv', index=False)
