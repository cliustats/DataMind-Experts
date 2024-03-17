# Import your libraries
import pandas as pd

# Start writing code
sat_scores.head()

sat_scores[sat_scores['sat_writing'] == sat_scores['sat_writing'].median()][['student_id']]
