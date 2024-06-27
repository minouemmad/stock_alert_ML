import pandas as pd
import psycopg2
import os

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
        date,
        exec_name,
        transaction_type,
        price_per_share,
        total_value,
        shares_following,
        percent_change,
        transaction_code
    FROM processed_listings
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    data = fetch_data()
    data.to_csv('insider_trading_data.csv', index=False)
