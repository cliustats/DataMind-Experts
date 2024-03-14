# Import your libraries
import pandas as pd

# Start writing code
hotel_reviews.head()
hotel_arena_reviews = hotel_reviews[hotel_reviews['hotel_name'] == 'Hotel Arena']

# result = hotel_arena_reviews.groupby(['reviewer_score', 'hotel_name'])['review_date'].count().to_frame('n_reviews').reset_index().sort_values(['reviewer_score'])

result = hotel_arena_reviews.groupby(['reviewer_score','hotel_name']).size().to_frame('n_reviews').reset_index()
