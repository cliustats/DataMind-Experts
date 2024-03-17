# Import your libraries
import pandas as pd

# Start writing code
google_file_store.head()

# draft = google_file_store[google_file_store['filename'].str.contains('draft')]
# result = draft.contents.str.split('\W+', expand=True).stack().value_counts().reset_index()


df = google_file_store[google_file_store['filename'].str.contains('draft')]
df = df['contents'].str.split("\W+").explode().to_frame()
df.value_counts()
