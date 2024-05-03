# Import your libraries
import pandas as pd

# Start writing code
# worker.head()

title_worker_id = title.rename(columns={'worker_ref_id': 'worker_id'})
merged_df = pd.merge(worker, title_worker_id, on='worker_id')
max_salary = merged_df[merged_df['salary'] == merged_df['salary'].max()]
result = max_salary[['worker_title']].rename(columns={"worker_title": "best_paid_title"})
result


# SQL solution
SELECT * FROM worker_id
SELECT * FROM worker_id
   
