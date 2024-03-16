# Import your libraries
import pandas as pd

# Start writing code
fb_friend_requests.head()
# sent_requests = fb_friend_requests[fb_friend_requests['action']=='sent']
# accepted_requests = fb_friend_requests[fb_friend_requests['action']=='accepted']
# merged_df = pd.merge(sent_requests, accepted_requests, how='left', on=['user_id_sender', 'user_id_receiver'], suffixes = ['_sent', '_accepted'])
# merged_df = merged_df[['date_sent', 'action_sent', 'action_accepted']]
# merged_df.loc[merged_df['action_sent'] == 'sent', 'sent_count'] = 1
# merged_df.loc[merged_df['action_accepted'] == 'accepted', 'accepted_count'] = 1
# result = merged_df.groupby('date_sent')[['sent_count', 'accepted_count']].sum().reset_index()

# result['acceptance_rate'] = result['accepted_count'] / result['sent_count']
# result = result.rename(columns = {'date_sent': 'date_x'})
# result[['date_x', 'acceptance_rate']]

sent_requests = fb_friend_requests[fb_friend_requests['action']=='sent']
accepted_requests = fb_friend_requests[fb_friend_requests['action']=='accepted']
merged_df = pd.merge(sent_requests, accepted_requests, how='left', on=['user_id_sender', 'user_id_receiver'], suffixes = ['_sent', '_accepted'])

accepted_count = merged_df.groupby(["date_sent"]).count().reset_index()
accepted_count["acceptance_rate"] = accepted_count["action_accepted"]/accepted_count["action_sent"]
result = accepted_count.rename(columns = {'date_sent': 'date_x'})
result[['date_x', 'acceptance_rate']]
