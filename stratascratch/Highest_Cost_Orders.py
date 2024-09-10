# Import your libraries
import pandas as pd

# Start writing code
orders['order_date'] = orders['order_date'].dt.date
orders = orders[(orders['order_date'] < pd.to_datetime('2019-05-01')) &
                (orders['order_date'] > pd.to_datetime('2019-02-01'))]
df = pd.merge(orders, customers, left_on='cust_id', right_on='id')
result = df.groupby(['first_name', 'order_date'])['total_order_cost'].sum().to_frame('max_cost').reset_index()
result.loc[result['max_cost'] == result['max_cost'].max()]




df = pd.merge(customer, order, left_on='cust_id', right_on='id')
df['order_date'] = df['order_date'].dt.strftime('%Y-%m-%d')
df.groupby(['first_name', 'order_date'])['total_order_cost'].sum().reset_index() \
    .nlargest(1, 'total_order_cost', keep='all').rename(columns={'total_order_cost': 'max_cost'})
