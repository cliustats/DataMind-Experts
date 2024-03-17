# Import your libraries
import pandas as pd

# Start writing code
voting_results.head()
df = voting_results.dropna()
df2 = df.groupby('voter')['candidate'].count().to_frame('number_of_people_voted').reset_index()
df2['value'] = round(1 / df2['number_of_people_voted'], 3)
df3 = pd.merge(df, df2, on='voter', how='left')
df4 = df3.groupby('candidate')['value'].sum().reset_index()
df4['rank'] = df4['value'].rank(method='dense', ascending=False)
df4[df4['rank'] == 1]['candidate']
