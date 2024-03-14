# Import your libraries
import pandas as pd

# Start writing code
merged_df = pd.merge(airbnb_units, airbnb_hosts, on='host_id')
filtered_df = merged_df[(merged_df['age'] < 30) & (merged_df['unit_type'] == 'Apartment')]
result = filtered_df.groupby('nationality')['unit_id'].nunique().to_frame('apartment_count').reset_index().sort_values('apartment_count', ascending=False)
