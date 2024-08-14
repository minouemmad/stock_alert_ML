import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, df):

    # Preserve the target variable
    if 'log_price' not in df.columns:
        raise ValueError("The target column 'log_price' is missing from the DataFrame.")
    
    # Isolate the target variable before any transformations
    y = df['log_price'].copy()

    # Handle categorical variables by ensuring the same one-hot encoding as in training
    df = pd.get_dummies(df.drop(columns=['log_price']), drop_first=True)

    # Replace infinite values with NaN and fill NaNs
    df.replace([float('inf'), float('-inf')], float('nan'), inplace=True)
    df.fillna(df.mean(), inplace=True)

    # Ensure the model and evaluation data have the same features
    model_features = model.feature_names_in_
    missing_cols = set(model_features) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    # Reorder columns to match the model's expectations
    df = df[model_features]

    # X remains the features
    X = df

    # Make predictions
    predictions = model.predict(X)

    # Calculate evaluation metrics
    mse = mean_squared_error(y, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y, predictions)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')

    # Visualization: Scatter plot of True vs Predicted Prices
    plt.scatter(y, predictions)
    plt.xlabel('True Prices')
    plt.ylabel('Predicted Prices')
    plt.title('True vs Predicted Prices')
    plt.show()

if __name__ == "__main__":
    # Load the processed feature data
    data = pd.read_csv('data\\processed\\features.csv')
    
    # Load the trained model using joblib
    model = joblib.load('models\\stock_prediction_model.pkl')
    
    # Evaluate the model
    evaluate_model(model, data)
