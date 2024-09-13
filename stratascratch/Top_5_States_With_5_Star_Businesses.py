# Import your libraries
import pandas as pd

# Start writing code
yelp_business.head()

five_star_business = yelp_business[yelp_business['stars'] == 5]
five_star_business = five_star_business.groupby('state')['business_id'].count().to_frame('n_businesses').reset_index()
five_star_business['rank'] = five_star_business['n_businesses'].rank(method='min', ascending=False)
result = five_star_business[five_star_business['rank'] <= 5][['state', 'n_businesses']].sort_values(['state', 'n_businesses'], ascending=[False, True])




yelp_business.head()


yelp_business[yelp_business['stars'] == 5].groupby('state')['business_id'].count().reset_index().rename(columns={'business_id': 'n_businesses'}).sort_values(['n_businesses', 'state'], ascending=[False, True]).nlargest(5, 'n_businesses', keep='all')
