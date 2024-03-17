# Import your libraries
import pandas as pd

# Start writing code
orders.head()

orders['order_month'] = orders['order_date'].dt.month
march_orders = orders[orders['order_month'] == 3]
result = march_orders.groupby('cust_id')['total_order_cost'].sum().to_frame('revenue').reset_index()
result.sort_values('revenue', ascending=False)
