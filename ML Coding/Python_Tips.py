################################################################
#                        Default Parameter Settings
################################################################
class Player:
    def __init__(self, name, items=[])
            self.name = name
            self.items = items
            # print(id(self.items))

p1 = Player('Alice')
p2 = Player('Bob')
p3 = Player('Charles', ['sword'])

p1.items.append('armor')
p2.items.append('sword')

print(p1.items)
# Returns ['armor', 'sword']

# The code as written contains a mistake: the default argument for items is a mutable object ([]).
# This can lead to unexpected behavior because mutable default arguments are shared across all instances
#  of the class that don't explicitly pass a value for that argument.


----- Correct way -------
class Player:
    def __init__(self, name, items=None)
            self.name = name
            self.items = items if items is not None else []

            # if items is None:
            #     self.items = []
            # else:
            #     self.items = items
            # print(id(self.items))

################################################################
#                        is None
################################################################
'''
Tip 2: When dealing with None, the best practice is to use the is operator for comparisons with None
'''
if a:
  print('Not None')
# This checks whether a is truthy. In Python, None, 0, False, empty collections (e.g., [], {}), and empty strings ("") evaluate to False.
# Drawback: This approach doesn’t specifically test for None. It might give misleading results if a is something else that evaluates to False (e.g., 0 or []).

if a == None:
  print('None')
# This works because None is equal to itself.
# Drawback: It’s less idiomatic and might be prone to issues if custom objects override the == operator in a way that makes them equal to None.
if a is None:
  print('None')
# This is the best approach for checking if a variable is None.
# The is operator checks identity—whether a is the same object as None.
# Why it's better: It’s explicit, Pythonic, and doesn’t rely on potentially overridden equality methods.

################################################################
#                        Decorator
################################################################









####  Missing values
df.isna().sum()

# mean imputation, frequency imputation, target imputation
mode_city = df['City'].mode()[0]
df['City'] = df['City'].fillna(mode_city)

# Impute missing values in 'Feature' based on the mean grouped by 'Target'
df['Feature'] = df.groupby('Target')['Feature'].apply(
    lambda x: x.fillna(x.mean())
)

df.duplicated().sum()

df.describe(numeric_only=True)
df.describe(include='object')
df.describe(include='all')

# Outlier Detection
def detect_outlier_iqr(df):
    outlier_count = {}
    for column in df.select_dtypes(include='number'):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR 
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outlier_count[column] = outliers.shape[0]

    return outlier_count

outliers_iqr = detect_outlier_iqr(df)
outlier_df = pd.DataFrame(
    list(outliers_iqr.item()), columns=['Column', 'Number of Outliers']
)
outlier_df
