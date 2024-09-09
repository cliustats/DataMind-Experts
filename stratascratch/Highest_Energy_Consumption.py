# Import your libraries
import pandas as pd

# Start writing code
fb_eu_energy.head()
df = pd.concat([fb_eu_energy, fb_asia_energy, fb_na_energy])
result = df.groupby('date')['consumption'].sum().to_frame('total_consumption').reset_index()
# result.sort_values('total_consumption', ascending=False)
result.loc[result['total_consumption']==result['total_consumption'].max()]




# Import your libraries
import pandas as pd

# Start writing code
df = pd.concat([fb_eu_energy,fb_asia_energy,fb_na_energy])

result = df.groupby('date')['consumption'].sum().reset_index()
result[result['consumption']==result['consumption'].max()][['date','consumption']]
