# Import your libraries
import pandas as pd

# Start writing code
# facebook_friends.head()
fb = facebook_friends
fb2 = fb.rename(columns={'user1': 'user2', 'user2': 'user1'})
df = pd.concat([fb, fb2])
popularity = df.groupby('user1')['user2'].count().to_frame('num_of_friends').reset_index()
popularity['popularity_percent'] = 100 * popularity['num_of_friends'] / popularity['user1'].count()
popularity[['user1', 'popularity_percent']]
