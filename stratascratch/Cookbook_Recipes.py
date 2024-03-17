# Import your libraries
import pandas as pd

# Start writing code
series = pd.DataFrame({'page_number': range(max(cookbook_titles.page_number) + 1)})
df = series.merge(cookbook_titles, how='left', on='page_number')
df['row_num'] = df.page_number // 2
df_left = df[df.page_number % 2 == 0].groupby('row_num').first().reset_index()
df_right = df[df.page_number % 2 == 1].groupby('row_num').first().reset_index()
result = df_left.merge(df_right, how='outer', on='row_num').sort_values(by='row_num')[['page_number_x','title_x', 'title_y']].rename(columns={'page_number_x': 'left_page_number', 'title_x': 'left_title', 'title_y': 'right_title'})
