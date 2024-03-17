# Import your libraries
import pandas as pd

# Start writing code
yelp_business.head()

yelp_business['categories'] = yelp_business['categories'].str.split(';')
yelp_business = yelp_business.explode('categories')
result = yelp_business.groupby('categories')['review_count'].sum().to_frame('total_reviews').reset_index()
result.sort_values('total_reviews', ascending=False)
