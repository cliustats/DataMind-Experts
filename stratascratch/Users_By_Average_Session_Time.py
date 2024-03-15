# Import your libraries
import pandas as pd
# Start writing code
facebook_web_log.head(20)

loads = facebook_web_log.loc[facebook_web_log['action']=='page_load', ['user_id', 'timestamp']]
exits = facebook_web_log.loc[facebook_web_log['action']== 'page_exit', ['user_id', 'timestamp']]

session = pd.merge(loads, exits, on='user_id', how='inner', suffixes=['_load', '_exit'])

session = session[session['timestamp_load'] < session['timestamp_exit']]

session['date_load'] = session['timestamp_load'].dt.date
session = session.groupby(['user_id', 'date_load']).agg({'timestamp_load': max, 'timestamp_exit': min}).reset_index()

session['duration'] = session['timestamp_exit'] - session['timestamp_load']

session.groupby('user_id')['duration'].mean().reset_index()
