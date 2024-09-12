# Import your libraries
import pandas as pd

# Start writing code
fact_events.head()

events_list = ['video call received', 'video call sent', 'voice call received', 'voice call sent']
fact_events['event_check'] = fact_events['event_type'].apply(lambda x: 1 if x in events_list else 0)
fact_events['event_check_mean'] = fact_events.groupby('user_id')['event_check'].transform('mean')

result = fact_events[fact_events['event_check_mean']>=0.5].groupby('client_id')['id'].count().reset_index()
result['ranking'] = result['id'].rank(ascending=False)
result = result[result.ranking == 1][['client_id']]
