# Import your libraries
import pandas as pd

# Start writing code
ms_user_dimension.head()

df = pd.merge(ms_download_facts, ms_user_dimension, on='user_id', how='left')
df2 = pd.merge(df,ms_acc_dimension , on='acc_id')
#df2.groupby(['date', 'paying_customer'])['downloads'].sum().reset_index()

df_new = df2.pivot_table(index='date', columns='paying_customer',
                         values='downloads', aggfunc='sum').reset_index()
df_new[df_new['no'] > df_new['yes']]
