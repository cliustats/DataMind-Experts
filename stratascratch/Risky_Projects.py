# Import your libraries
import pandas as pd
import numpy as np
# Start writing code
# linkedin_projects.head()
# linkedin_emp_projects.groupby('project_id').size().reset_index()
# linkedin_employees.head()
linkedin_projects = linkedin_projects.rename(columns={'id': 'project_id'})
linkedin_employees = linkedin_employees.rename(columns={'id': 'emp_id'})
df = pd.merge(linkedin_projects, linkedin_emp_projects, on='project_id')
df = pd.merge(df, linkedin_employees, on='emp_id')
df['duration_prop'] = (df['end_date'] - df['start_date']).dt.days / 365
df['prorated_salary'] = df['salary'] * df['duration_prop']
result = df.groupby(['title', 'budget'])['prorated_salary'].sum().to_frame('prorated_expense').reset_index()

result['prorated_expense'] = result['prorated_expense'].apply(np.ceil)
result.loc[result['budget'] < result['prorated_expense']].drop_duplicates().sort_values('title')
