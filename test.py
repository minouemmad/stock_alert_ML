import pandas as pd
import psycopg2
import os

conn = psycopg2.connect(
        host=os.environ['POSTGRES_ENDPOINT'],
        database='stock_alert',
        user='readonly',
        password=os.environ['READONLY_PASSWORD'],
        port=5432,
        sslmode='verify-ca',
        sslrootcert='./global-bundle.pem'
    )
cur = conn.cursor()
cur.execute("""select * from processed_listings limit 2000""")
df = pd.DataFrame(cur.fetchall())
print(df)
conn.commit()
conn.close()