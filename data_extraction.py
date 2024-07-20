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
    LIMIT 500
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == "__main__":
    data = fetch_data()
    data.to_csv('insider_trading_data.csv', index=False)
