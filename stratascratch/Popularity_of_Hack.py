# Import your libraries
import pandas as pd

# Start writing code
# facebook_employees.location.value_counts().reset_index()

facebook_hack_survey.head()

df = pd.merge(facebook_hack_survey, facebook_employees, left_on='employee_id', right_on='id', how='left')
df.groupby('location')['popularity'].mean().reset_index()
