import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate_model(model, df):
    X = df.drop(columns=['date', 'price_per_share', 'log_price'])
    y = df['log_price']

    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f'Mean Squared Error: {mse}')

    plt.scatter(y, predictions)
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title('True vs Predicted Prices')
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('features.csv')
    model = XGBRegressor()
    model.load_model('stock_price_model.json')
    evaluate_model(model, data)
