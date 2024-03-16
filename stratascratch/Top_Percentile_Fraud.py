# Import your libraries
import pandas as pd

# Start writing code
fraud_score.head()

fraud_score['rank'] = fraud_score.groupby('state')['fraud_score'].rank(pct=True)
top_5_pct_claims = fraud_score[fraud_score['rank'] > 0.95]
top_5_pct_claims.loc[:, top_5_pct_claims.columns.drop('rank')]
