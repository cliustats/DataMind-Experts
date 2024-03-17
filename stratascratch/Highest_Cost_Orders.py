# Import your libraries
import pandas as pd

# Start writing code
orders['order_date'] = orders['order_date'].dt.date
orders = orders[(orders['order_date'] < pd.to_datetime('2019-05-01')) &
                (orders['order_date'] > pd.to_datetime('2019-02-01'))]
df = pd.merge(orders, customers, left_on='cust_id', right_on='id')
result = df.groupby(['first_name', 'order_date'])['total_order_cost'].sum().to_frame('max_cost').reset_index()
result.loc[result['max_cost'] == result['max_cost'].max()]
