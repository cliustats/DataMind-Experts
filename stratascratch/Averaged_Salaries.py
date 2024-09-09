# Import your libraries
import pandas as pd

# Start writing code
employee.head()

df = employee.groupby('department')['salary'].mean().reset_index()
df = df.rename(columns = {'salary': 'avg_salary'})
result = pd.merge(employee, df, on='department', how='left')
result[['department', 'first_name', 'salary', 'avg_salary']]
