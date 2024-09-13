### Write a query that'll identify returning active users. A returning active user is a user that has made
### a second purchase within 7 days of any other of their purchases.
### Output a list of user_ids of these returning active users.


# Import your libraries
import pandas as pd

# Start writing code
# amazon_transactions = amazon_transactions.sort_values(['user_id', 'created_at'])

# amazon_transactions['created_at_date'] = amazon_transactions['created_at'].dt.date
# amazon_transactions['previous_date'] = amazon_transactions.groupby('user_id')['created_at_date'].shift()
# amazon_transactions['days'] = (amazon_transactions['created_at_date'] - amazon_transactions['previous_date']).dt.days
# result = amazon_transactions[amazon_transactions['days'] <= 7]['user_id'].unique()

df = amazon_transactions.sort_values(['user_id','created_at'])
df['diff'] = df.groupby('user_id')['created_at'].diff().dt.days
result = df[df['diff'] <= 7]['user_id'].unique()
result





df = amazon_transactions.sort_values(['user_id', 'created_at'])

df['next_purchase_time'] = df.groupby('user_id')['created_at'].diff()
df[df['next_purchase_time'] <= pd.Timedelta(days=7)]['user_id'].unique()
