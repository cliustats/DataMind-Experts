# Import your libraries
import pandas as pd

# Start writing code
google_file_store.head()
df = google_file_store['contents'].str.lower().str.split().explode().value_counts().reset_index()
df[(df['index']=='bull') | (df['index']=='bear')]
