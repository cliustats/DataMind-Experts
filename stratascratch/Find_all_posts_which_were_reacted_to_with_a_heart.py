# Import your libraries
import pandas as pd

# Start writing code
# post_with_reactions = pd.merge(facebook_posts, facebook_reactions, on=['post_id', 'poster'])
# hearted_post = post_with_reactions[post_with_reactions['reaction'] == 'heart']
# hearted_post[facebook_posts.columns].drop_duplicates('post_id')


# # filter before merge, and only keep useful information. In this case, the post_id
heart = facebook_reactions[facebook_reactions['reaction'] == 'heart'][['post_id']]
result = pd.merge(heart, facebook_posts, on='post_id').drop_duplicates('post_id')
