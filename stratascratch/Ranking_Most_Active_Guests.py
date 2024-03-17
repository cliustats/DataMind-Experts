# Import your libraries
import pandas as pd

# Start writing code
airbnb_contacts.head()
df = airbnb_contacts.groupby('id_guest')['n_messages'].sum().to_frame('n_messages').reset_index()
df['rank'] = df['n_messages'].rank(ascending=False, method='dense')
result = df.sort_values('rank')
