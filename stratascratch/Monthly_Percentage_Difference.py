# Import your libraries
import pandas as pd

# Start writing code
sf_transactions.head()
sf_transactions['year_month'] = sf_transactions['created_at'].dt.strftime('%Y-%m')
df = sf_transactions.groupby('year_month')['value'].sum().to_frame('month_revenue').reset_index()
df['last_month_revenue'] = df['month_revenue'].shift(1)
df['revenue_diff_pct'] = 100.0 * ((df['month_revenue'] - df['last_month_revenue']) / df['last_month_revenue']).round(4)
df[['year_month', 'revenue_diff_pct']]
