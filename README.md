# Insider Trading Stock Prediction Project Plan

This document outlines the phases of the insider trading stock prediction project. Each phase is designed to ensure a structured and systematic approach to building a machine learning model that predicts stock price movements based on insider trading activities. This project will contribute to Carlo Tran's Stock Alert app by providing enhanced predictive capabilities.

## Project Phases

### 1. Data Extraction

**Goal:** Retrieve and preprocess relevant insider trading data from the PostgreSQL database.

**Actions:**
- **Database Connection:** Establish a secure connection to the PostgreSQL database where insider trading data is stored.
- **Data Retrieval:** Execute SQL queries to fetch required data fields, such as transaction details, stock information, and insider profiles.
- **Data Storage:** Save the extracted data into a CSV file for further processing, ensuring that the file structure aligns with downstream tasks like feature engineering and model training.

### 2. Feature Engineering

**Goal:** Transform raw insider trading data into meaningful, predictive features that capture the nuances of stock price movements.

**Actions:**
- **Date Feature Extraction:** Convert date fields into datetime objects and extract relevant components such as year, month, day, and day of the week. This allows the model to identify temporal patterns in insider trading activities.
  
- **Transaction-Related Features:** 
  - **Transaction Type:** Categorize each transaction as a purchase or sale to understand the intent behind the trade.
  - **Price Per Share & Total Value:** Calculate these metrics to quantify the financial magnitude of each transaction.
  - **Shares Following Transactions:** Capture the number of shares held by the insider after the transaction, providing context to the insider's confidence in the company's future.
  - **Percent Change in Shares:** Reflect the magnitude of the trade relative to the insider's previous holdings. This feature is crucial as large percentage changes might indicate significant confidence or concern, potentially leading to stock price fluctuations.
  - **Lagged Features:** Incorporate previous transaction prices (e.g., price_lag_1, price_lag_2) to model the impact of past insider trades on future stock prices.

- **Insider Profile Features:** 
  - **Owner Type:** Include information about the insider's role (e.g., CEO, CFO, 10% owner). Different roles carry different levels of influence over a company's operations, and their trades can have varying impacts on stock prices.
  
- **Industry Context Features:** 
  - **Division, Major Group, Industry Group, SIC:** Describe the industry sector the company belongs to, offering critical context for stock price movements. These features help the model account for broader industry trends that could affect stock performance, independent of individual insider trades.

- **Normalization & Scaling:** Apply appropriate normalization or scaling techniques to numerical features to ensure they are on a similar scale, which is essential for models like XGBoost that are sensitive to feature magnitude.

- **Categorical Variable Handling:** Implement techniques like one-hot encoding or label encoding to handle categorical variables, ensuring they are in a format suitable for model consumption.

### 3. Model Training

**Goal:** Develop and train a robust machine learning model using the engineered features to predict stock price movements.

**Actions:**
- **Data Splitting:** Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.
- **Algorithm Selection:** Choose a suitable machine learning algorithm, such as XGBoost or Random Forest, that can handle the complexities of the dataset, including non-linear relationships and interactions between features.
- **Model Training:** Train the model using the training data, optimizing it to capture patterns that link insider trading activities with subsequent stock price movements.
- **Hyperparameter Optimization:** Use techniques like cross-validation to fine-tune hyperparameters, enhancing the model's performance and generalization capability.
- **Performance Evaluation:** Assess the model's performance on the test set using metrics like Mean Squared Error (MSE) to ensure it can accurately predict stock prices.
- **Model Persistence:** Save the trained model as a serialized object for future use in prediction tasks.

### 4. Model Evaluation

**Goal:** Rigorously evaluate the model's performance and validate its predictive capabilities on unseen data.

**Actions:**
- **Model Loading:** Load the trained model from storage.
- **Prediction:** Make predictions on the test dataset and compare them with actual stock prices.
- **Evaluation Metrics:** Calculate evaluation metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared to quantify the model's accuracy.
- **Visualization:** Create scatter plots of predicted vs. actual values to visually assess the model's accuracy and identify potential biases.
- **Iteration:** Based on evaluation results, iterate on the model design by adjusting features, trying different algorithms, or refining hyperparameters to improve performance.

### 5. Deployment and Predictions

**Goal:** Deploy the trained model to make real-time predictions on new insider trading data, contributing to Carlo Tran's Stock Alert app.

**Actions:**
- **Data Preparation:** Prepare new insider trading data for prediction by applying the same feature engineering steps used during model training.
- **Model Deployment:** Load the trained model into a production environment where it can be accessed for real-time predictions.
- **User Interface Integration:** Implement an interface to accept new data inputs and produce real-time stock price predictions, integrating seamlessly with the Stock Alert app.
- **Monitoring & Maintenance:** Continuously monitor the model's performance in production, retraining or updating it as necessary to maintain accuracy and relevance.

## Dependencies

- Python 3.8+
- pandas
- psycopg2
- scikit-learn
- XGBoost (optional, depending on model choice)
- matplotlib (for visualization)
