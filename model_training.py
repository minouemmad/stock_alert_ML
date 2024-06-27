import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def train_model(df):
    X = df.drop(columns=['date', 'price_per_share', 'log_price'])
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    return model

if __name__ == "__main__":
    data = pd.read_csv('features.csv')
    model = train_model(data)
    model.save_model('stock_price_model.json')
