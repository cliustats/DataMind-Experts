# Import your libraries
import pandas as pd

# Start writing code
user_flags.head()
approved_flags = flag_review[flag_review['reviewed_outcome']=='APPROVED']
df = pd.merge(approved_flags, user_flags, on='flag_id')
df['username'] = df['user_firstname'] + ' ' + df['user_lastname']
result = df.groupby('username')['video_id'].nunique().to_frame('distinct_videos').reset_index()
result[result['distinct_videos']==result['distinct_videos'].max()]['username']
