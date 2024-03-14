# Start writing code
dc_bikeshare_q1_2012.head()

# method 1
# sorted_dc_bikeshare = dc_bikeshare_q1_2012.sort_values('end_time', ascending=False)
# latest_dc_bikeshare = sorted_dc_bikeshare.drop_duplicates('bike_number')
# result = latest_dc_bikeshare.sort_values('end_time' , ascending=False)[['bike_number', 'end_time']]
# result.rename(columns={'end_time': 'last_used'})

# method 2
# result = dc_bikeshare_q1_2012.groupby('bike_number')['end_time'].max().reset_index().sort_values('end_time', ascending=False)
# result.rename(columns={'end_time': 'last_used'})

# method 3
result = dc_bikeshare_q1_2012.groupby('bike_number')['end_time'].max().to_frame('last_used').reset_index().sort_values('last_used', ascending=False)
