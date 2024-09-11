# Import your libraries
import pandas as pd

# Start writing code
yelp_reviews.head(10)
yelp_reviews[yelp_reviews['cool'] == yelp_reviews['cool'].max()][['business_name', 'review_text']]
