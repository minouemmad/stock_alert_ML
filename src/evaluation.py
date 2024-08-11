#evaluation.py
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def evaluate_model(model, df):
    print("Columns in the DataFrame:", df.columns.tolist())

    X = df.drop(columns=['date', 'percent_change_in_shares', 'log_price'])
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
    # Load the processed feature data
    data = pd.read_csv('C:\\Users\\maemm\\OneDrive\\Desktop\\GitHub\\stock_alert_ML\\data\\processed\\features.csv')
    
    # Load the trained model using joblib
    model = joblib.load('C:\\Users\\maemm\\OneDrive\\Desktop\\GitHub\\stock_alert_ML\\models\\stock_prediction_model.pkl')
    
    # Evaluate the model
    evaluate_model(model, data)
