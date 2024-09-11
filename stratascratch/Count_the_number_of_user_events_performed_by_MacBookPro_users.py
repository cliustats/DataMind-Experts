# Import your libraries
import pandas as pd

# Start writing code
playbook_events.head()
macbook_events = playbook_events[playbook_events['device'] == 'macbook pro']
macbook_events.groupby('event_name')['user_id'].count().reset_index()
