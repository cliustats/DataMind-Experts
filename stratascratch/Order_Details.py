# Import your libraries
import pandas as pd

# Start writing code
customers.head()

df = pd.merge(orders, customers, left_on='cust_id', right_on='id')
df[df['first_name'].isin(['Jill', 'Eva'])].sort_values('cust_id')[["first_name", "order_date", "order_details", "total_order_cost"]]
