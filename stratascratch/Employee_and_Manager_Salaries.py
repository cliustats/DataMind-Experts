# Import your libraries
import pandas as pd

# Start writing code
employee.head()

df = pd.merge(employee, employee, left_on='manager_id', right_on='id', how='left', suffixes=(['_employer', '_manager']))
df[df['salary_employer'] > df['salary_manager']][['first_name_employer', 'salary_employer']] \
    .rename(columns = {'first_name_employer': 'first_name', 'salary_employer': 'salary'})
