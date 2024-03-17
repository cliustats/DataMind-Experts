# Import your libraries
import pandas as pd

# Start writing code
# facebook_posts.head()
# viewed_posts = pd.merge(facebook_posts, facebook_post_views, on='post_id').drop_duplicates('post_id')
# viewed_posts['spam'] = viewed_posts['post_keywords'].str.contains('spam')
#
# # result = viewed_posts['spam'].sum() / viewed_posts['post_id'].count()
# result = viewed_posts.groupby('post_date').agg({'spam': ['sum', 'count']}).reset_index()
# result.columns = ['post_date', 'spam_sum', 'spam_count']
# result['spam_share'] = 100 * result['spam_sum'] / result['spam_count']
# result[['post_date', 'spam_share']]



viewed_posts = pd.merge(facebook_posts, facebook_post_views, on='post_id').drop_duplicates('post_id')
viewed_posts['spam'] = viewed_posts['post_keywords'].str.contains('spam')
result = viewed_posts.groupby('post_date').agg({'spam': 'sum', 'viewer_id': 'count'}).reset_index()
result['spam_share'] = 100 * result['spam'] / result['viewer_id']
result[['post_date', 'spam_share']]
