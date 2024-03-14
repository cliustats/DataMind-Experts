# Import your libraries
import pandas as pd

# Start writing code
oscar_nominees.head()

ab_movies = oscar_nominees[oscar_nominees['nominee'] == 'Abigail Breslin']
result = ab_movies['movie'].nunique()
