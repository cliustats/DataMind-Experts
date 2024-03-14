# Import your libraries
import pandas as pd

# Start writing code
ms_employee_salary.head()

# sorted_salaries = ms_employee_salary.sort_values('salary', ascending = False)
# current_salaries = sorted_salaries.drop_duplicates('id')
# result = current_salaries.sort_values('id')
# result

result = ms_employee_salary.groupby(['id', 'first_name', 'last_name', 'department_id'])['salary'].max().reset_index().sort_values('id')
