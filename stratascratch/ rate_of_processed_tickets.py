# Import your libraries
import pandas as pd

# Start writing code
facebook_complaints.head()

# method 1
# df = facebook_complaints.groupby('type').agg({'processed':['sum', 'count']}).reset_index()

# df['processed_rate'] = df['processedsum'] / df['processedcount']
# df[['type', 'processed_rate']]

# method 2
df = facebook_complaints.groupby('type').agg(processed_sum = ('processed', 'sum'),
                                             processed_count = ('processed', 'count')).reset_index()
df['processed_rate'] = df['processed_sum'] / df['processed_count']
df[['type', 'processed_rate']]



# method 3
facebook_complaints['processed'] = facebook_complaints['processed'].astype(int)
grouped = facebook_complaints.groupby(['type']).agg({'processed':'sum','complaint_id':'size'}).reset_index()
grouped['processed_rate'] =grouped['processed']/grouped['complaint_id']
result = grouped[['type','processed_rate']]
