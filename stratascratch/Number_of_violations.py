# Import your libraries
import pandas as pd

# Start writing code
sf_restaurant_health_violations.head()
sf = sf_restaurant_health_violations
rox_cafe = sf[sf['business_name']=='Roxanne Cafe' ]
# rox_cafe = rox_cafe[rox_cafe['violation_id'].notnull()]
rox_cafe['inspection_date'] = rox_cafe['inspection_date'].dt.year
rox_cafe.groupby('inspection_date')['violation_id'].count().reset_index().sort_values('inspection_date')
