# Import your libraries
import pandas as pd

# Start writing code
employee.head()

distinct_salary = employee.drop_duplicates(subset = 'salary')
distinct_salary['rnk'] = distinct_salary['salary'].rank(method='dense', ascending=False)
result = distinct_salary[distinct_salary.rnk == 2][['salary']]
