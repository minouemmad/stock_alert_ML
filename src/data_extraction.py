import pandas as pd
import psycopg2
import os
from sklearn.model_selection import train_test_split

def fetch_data():
    conn = psycopg2.connect(
        host=os.environ['POSTGRES_ENDPOINT'],
        database='stock_alert',
        user='readonly',
        password=os.environ['READONLY_PASSWORD'],
        port=5432,
        sslmode='verify-ca',
        sslrootcert='./global-bundle.pem'
    )
    
    query = """
    SELECT 
        trade_date,
        ticker,
        company_name,
        person_name,
        owner_type,
        num_shares,
        price_per_share,
        total_value_of_trade,
        shares_owned_following_transaction,
        percent_change_in_shares,
        transaction_codes,
        division,
        major_group
    FROM processed_listings
    ORDER BY trade_date DESC
    LIMIT 500000
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
    
    return train_df, test_df

if __name__ == "__main__":
    train_data, test_data = fetch_data()
    
    # Save training and testing data to Pickle format
    train_data.to_pickle('data\\raw\\insider_trading_train.pkl')
    test_data.to_pickle('data\\raw\\insider_trading_test.pkl')
    print("Training and testing data saved to Pickle format.")
