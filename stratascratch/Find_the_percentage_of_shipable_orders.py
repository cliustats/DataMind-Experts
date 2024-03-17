# Import your libraries
import pandas as pd

# Start writing code
orders.head()
valid_customers = customers[customers['address'].notnull()]
valid_customers = valid_customers.rename(columns = {'id': 'cust_id'})
df = pd.merge(orders, valid_customers, on='cust_id', how='left')
result = 100 * df['address'].count() / df['cust_id'].count()
result
