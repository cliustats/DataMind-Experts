# Import your libraries
import pandas as pd

# Start writing code
customers.head()
df = pd.merge(customers, orders, left_on='id', right_on='cust_id', how='left')
df = df[['first_name', 'last_name', 'city', 'order_details']]
df.sort_values(['first_name', 'order_details'])
