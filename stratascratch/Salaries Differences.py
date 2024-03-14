# Import your libraries
import pandas as pd

# Start writing code
# db_employee.head()

db_dept = db_dept.rename(columns={'id': 'department_id'})
merged_df = pd.merge(db_employee, db_dept, on='department_id')
highest_salary = merged_df.groupby('department')['salary'].max().reset_index()
market_highest_salary = highest_salary.loc[highest_salary['department'] == 'marketing', 'salary'].values
engineering_highest_salary = highest_salary.loc[highest_salary['department'] == 'engineering', 'salary'].values

market_highest_salary - engineering_highest_salary
