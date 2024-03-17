# Import your libraries
import pandas as pd

# Start writing code
sf_employee.head()
sf_bonus = sf_bonus.groupby('worker_ref_id')['bonus'].sum().reset_index()

df = pd.merge(sf_employee, sf_bonus, left_on='id', right_on='worker_ref_id', how='right')
df['total_comp'] = df['salary'] + df['bonus']
result = df.groupby(['employee_title', 'sex'])['total_comp'].mean().to_frame('avg_total_comp').reset_index()
result
