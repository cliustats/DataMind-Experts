# import pandas as pd

# result = sf_restaurant_health_violations.iloc[:, :2]
# result['business_type'] = result['business_name'].apply(lambda x: 'school' if 'school' in x.lower() \
#     else 'restaurant' if 'restaurant' in x.lower() \
#     else 'cafe' if 'cafe' in x.lower() or 'coffee' in x.lower() or 'café' in x.lower()
#     else 'other')
# result = result[['business_name', 'business_type']].drop_duplicates()

# Import your libraries
import pandas as pd
import numpy as np
# Start writing code
sf_restaurant_health_violations.head()
sf = sf_restaurant_health_violations
sf = sf[['business_name']].drop_duplicates()
low = sf.business_name.str.lower()
sf['type'] = np.where(low.str.contains(r'(cafe|café|coffee)'), 'cafe', np.where(low.str.contains(r'(school)'), 'school', np.where(low.str.contains(r'(restaurant)'), 'restaurant', 'other')))
sf
