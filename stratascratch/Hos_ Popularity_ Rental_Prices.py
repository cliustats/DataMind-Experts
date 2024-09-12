# Import your libraries
import pandas as pd
import numpy as np
# Start writing code
airbnb_host_searches.head()
df = airbnb_host_searches
df['host_id'] = df['price'].map(str) + df['room_type'].map(str) + df['host_since'].map(str) + df['zipcode'].map(str)+ df['number_of_reviews'].map(str)
df = df[['host_id','number_of_reviews','price']].drop_duplicates()

df['host_popularity'] = df['number_of_reviews'].apply(lambda x:'New' if x<1 else 'Rising' if x<=5 else 'Trending Up' if x<=15 else 'Popular' if x<=40 else 'Hot')

result = df.groupby('host_popularity').agg(min_price=('price',min),avg_price = ('price',np.mean),max_price = ('price',max)).reset_index()



result = df.groupby('host_popularity')['price'].agg(['min', 'mean', 'max'])
