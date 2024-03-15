# Import your libraries
import pandas as pd

# Start writing code
forbes_global_2010_2014.head()

# sorted_df = forbes_global_2010_2014.sort_values('profits', ascending=False)

# results = sorted_df[['company', 'profits']].head(3)

result = forbes_global_2010_2014.groupby('company')['profits'].sum().reset_index().sort_values(by='profits', ascending=False)
result['rank'] = result['profits'].rank(method='min', ascending=False)
result = result[result['rank'] <= 3][['company', 'profits']]
