# Import your libraries
import pandas as pd

# Start writing code
car_launches.head()
df = car_launches.groupby(['company_name', 'year']).size().to_frame('total_products').reset_index()
df['net_new_products'] = df.groupby('company_name')['total_products'].diff()
result = df.loc[df['net_new_products'].notnull()][['company_name', 'net_new_products']]
result
