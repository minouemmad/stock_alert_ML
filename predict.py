import pandas as pd
from xgboost import XGBRegressor

def predict_new_data(new_data):
    model = XGBRegressor()
    model.load_model('stock_price_model.json')

    feature_data = engineer_features(new_data)
    X = feature_data.drop(columns=['date', 'price_per_share', 'log_price'])
    predictions = model.predict(X)
    return predictions

if __name__ == "__main__":
    new_data = pd.read_csv('new_insider_trading_data.csv')
    predictions = predict_new_data(new_data)
    print(predictions)
