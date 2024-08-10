# Insider Trading Stock Prediction Project Plan

This document outlines the phases of the insider trading stock prediction project. Each phase is designed to ensure a structured and systematic approach to building a machine learning model for stock price prediction.

## Project Phases

### 1. Data Extraction

**Goal:** Retrieve relevant insider trading data from the PostgreSQL database.

**Actions:**
- Connect to the PostgreSQL database.
- Execute SQL queries to fetch required data.
- Save the extracted data into a CSV file for further processing.

### 2. Feature Engineering

**Goal:** Transform the raw insider trading data into meaningful features for model training.

**Actions:**
- Convert date fields into datetime objects.
- Engineer transaction-related features such as transaction type (purchase or sale), price per share, total value, shares following transactions, and percent change in shares.
- Calculate additional metrics that may influence stock prices, such as transaction times (e.g., hour of the day) and lagged features (previous transaction prices).
- Normalize or scale numerical features as necessary.
- Handle categorical variables appropriately (e.g., one-hot encoding or label encoding).

### 3. Model Training

**Goal:** Develop and train a machine learning model using the engineered features.

**Actions:**
- Split the dataset into training and testing sets.
- Choose a suitable machine learning algorithm (e.g., XGBoost, Random Forest).
- Train the model using the training data.
- Optimize hyperparameters through techniques like cross-validation.
- Evaluate the model's performance using appropriate metrics (e.g., Mean Squared Error for regression tasks).
- Save the trained model for future use.

### 4. Model Evaluation

**Goal:** Assess the performance of the trained model and validate its predictive capabilities.

**Actions:**
- Load the trained model.
- Make predictions on the test dataset.
- Calculate evaluation metrics such as Mean Squared Error, Root Mean Squared Error, or R-squared.
- Visualize the predicted vs. actual values to understand model accuracy and potential biases.
- Iterate on the model design based on evaluation results, adjusting features or algorithms as needed.

### 5. Deployment and Predictions

**Goal:** Deploy the trained model to make predictions on new insider trading data.

**Actions:**
- Prepare new insider trading data for prediction (similar to the feature engineering process).
- Load the trained model into a production environment.
- Implement an interface to accept new data inputs and produce real-time predictions.
- Monitor model performance in production and implement updates or retraining as necessary.

## Dependencies

- Python 3.8+
- pandas
- psycopg2
- scikit-learn
- XGBoost (optional, depending on model choice)
- matplotlib (for visualization)

This project utilizes additional columns in the feature engineering process to capture more nuanced insights about insider trading activities:

owner_type: Provides insight into the role of the insider (e.g., CEO, CFO, 10% owner).
percent_change_in_shares: Reflects the magnitude of the trade relative to the insider's previous holdings.
division, major_group, industry_group, sic: Describe the industry sector the company belongs to, offering critical context for stock price movements.

This code is intended to be an extra feature to Carlo Tran's Stock Alert app.
