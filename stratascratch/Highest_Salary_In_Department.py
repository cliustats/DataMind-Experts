# Import your libraries
import pandas as pd

# Start writing code
employee.head()

highest = employee.groupby('department')['salary'].max().reset_index().rename(columns={'salary': 'max_salary'})
df = pd.merge(employee, highest, on='department')
df[df['salary'] == df['max_salary']][['department', 'first_name', 'salary']]
