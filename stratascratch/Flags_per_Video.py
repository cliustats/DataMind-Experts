# Import your libraries
import pandas as pd

# Start writing code
# user_flags.head()
user_flags['user_name'] = user_flags['user_firstname'].astype(str) + ' ' + user_flags['user_lastname'].astype(str)
user_flags = user_flags[user_flags['flag_id'].notnull()]
user_flags.groupby('video_id')['user_name'].nunique().to_frame('num_unique_users').reset_index()

# result = user_flags[user_flags["flag_id"].notnull()]
# result["username"] = result["user_firstname"].astype(str) + " " + result["user_lastname"].astype(str)
# result = result.groupby(by="video_id")["username"].nunique().reset_index()
# result = result.rename(columns={"username": "num_unique_users"})
