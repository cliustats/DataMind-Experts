# Import your libraries
import pandas as pd

# Start writing code
winemag_p1.head()

result = winemag_p1[winemag_p1['description'].str.lower().str.contains('(plum|cherry|rose|hazelnut)([^a-z])')]
result[['winery']].drop_duplicates()
