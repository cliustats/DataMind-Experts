# Import your libraries
import pandas as pd

# Start writing code
airbnb_search_details.head()

airbnb_search_details.groupby(['city', 'property_type'])[['bedrooms', 'bathrooms']].mean().reset_index().rename(columns = {'bedrooms': 'n_bedrooms_avg', 'bathrooms': 'n_bathrooms_avg'})
