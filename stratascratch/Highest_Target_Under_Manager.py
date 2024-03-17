# Import your libraries
import pandas as pd

# Start writing code
salesforce_employees.head()

df = salesforce_employees[salesforce_employees['manager_id'] == 13][['first_name', 'target']]
df[df['target'] == df['target'].max()]
