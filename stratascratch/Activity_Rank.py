# Import your libraries
import pandas as pd

# Start writing code
google_gmail_emails.head(10)
result = google_gmail_emails.groupby('from_user').size().to_frame('total_emails').reset_index()
result['rank'] = result['total_emails'].rank(method = 'first', ascending=False)
result = result.sort_values(['total_emails', 'from_user'], ascending=[False, True])
