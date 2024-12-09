1. When to Refresh an ML Model? Why model performance might differ in production vs in development?
      https://www.youtube.com/watch?v=w9ipP9kPpFc&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=22
2. ML Training Data vs. Testing Data  https://www.youtube.com/watch?v=_Y7E2YKfuFM&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=20
3. Handling Exploding Gradients in Machine Learning   https://www.youtube.com/watch?v=Y5DVYLmmdrw&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=19
  1) Clipping the gradidents
  2) batch normalization
  3) Change the architecture. reduce the hidden layers, or transformer with skip connections




# Debug ----- pdb


################################################################
                        Virtual Environment
################################################################

#!/bin/bash

mkdir

venv   # storing the things related to virtual environment

rm -r venv   # delete file

cd /usercode/FIELSYSTEM

# create an virtual environment
python3 -m venv env    # creates a new directory

# active the virtual environment
source env/bin/activate

pip list              # show the packages
pip install requests

deactivate      # deactivate the

pip freeze > requirements.txt

python -m venv my_env

my_env/bin/activate

pip install -r requirements.txt     ##  -r :  install from a file



################################################################
                        Decorator
################################################################

def log_wrapper(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__} with arguments: {args}, {kwargs}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper


def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"An error occurred: {e}")

    return wrapper


@error_handler
def divide(a, b):


################################################################
                             Class
################################################################

class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_grade(self):
        return self.grade


class Course:
    def __init__(self, name, max_students):
        self.name = name
        self.max_students = max_students
        self.students = []

    def add_student(self, student):
        if len(self.students) <= self.max_students:
            self.students.append(student)
            return True
        return False

    def get_average_grade(self):
        value = 0
        for student in self.students:
            value += student.get_grade()
        return value / len(self.students)


------------------------------

# Inheritance

class Pet:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show(self):
        print(f"I am {self.name} and I am {self.age} years old")

    def speak(self):
        print("I don't know what to say")


class Cat(Pet):
    def __init__(self, name, age, color):
        super().__init__(name, age)
        self.color = color

    def speak(self):
        print("Meow")


class Dog(Pet):
    def speak(self):
        print("Barl")


class Fish(Pet):
    pass


p = Pet("Tim", 19)
p.speak()

c = Cat("Bill", 34, "green")
c.speak()

f = Fish("bubble", 1)
f.speak()


################################################################
                             Numpy
################################################################

np_array = np.array([[1, 2, 3], [4, 5, 6]])

print("Dimensions: ", np_array.ndim) # Dimensions:  2
print("Shape: ", np_array.shape)     # Shape:  (2, 3)
print("Size: ", np_array.size)       # Size: 6
print("Data Type: ", np_array.dtype) # Data Type:  int64

reshaped_array = np_array.reshape(3, 2)

################################################################
                            Pandas
################################################################

import pandas as pd

data_dict = {"Name": ["John", "Anna", "Peter"],
             "Age": [28, 24, 33],
             "City": ["New York", "Los Angeles", "Berlin"]}

df = pd.DataFrame(data_dict)

print(df)

"""
    Name  Age         City
0   John   28     New York
1   Anna   24  Los Angeles
2  Peter   33       Berlin
"""

print(df.head(2))  # Print first two rows
print(df.tail(2))  # Print last two rows
print(df.shape)    # Print dimensions of the df (rows, columns): (3, 3)
print(df.columns)  # Print column labels: Index(['Name', 'Age', 'City'], dtype='object')
print(df.dtypes)   # Print data types of each column:
# Name    object
# Age      int64
# City    object
# dtype: object


df["isYouthful"] = df["Age"].apply(lambda age: "Yes" if age < 30 else "No")


df2 = pd.DataFrame({"Name": ["Megan"], "Age": [34], "City": ["San Francisco"], "IsYouthful": ["No"]})

# pd.concat: default is axis=0, meaning rows are concatenated
df_concatenated = pd.concat([df, df2], ignore_index=True)
# Column-Wise Concatenation:
pd.concat([df, df2], axis=1)
# Handling Overlapping Columns: Use join='inner' to only keep overlapping columns:
pd.concat([df, df2], join='inner')


print(df['column_name']) # select a single column
print(df[['col1', 'col2']]) # select multiple columns

df.loc (Label-Based Indexing)
df.iloc (Integer-Based Indexing)


################################################################
                    Descriptive Statistics
################################################################

import numpy as np
import pandas as pd
import seaborn as sns

# Load Titanic dataset
titanic_df = sns.load_dataset('titanic')

mean_age = titanic_df['age'].mean()
median_age = titanic_df['age'].median()
mode_age = titanic_df['age'].mode()[0]

std_dev_age = np.std(titanic_df['age'])

# Quartiles and percentiles
# Using Numpy
Q1_age_np = np.percentile(titanic_df['age'].dropna(), 25) # dropna is being used to drop NA values

# Using Pandas
Q1_age_pd = titanic_df['age'].quantile(0.25)


################################################################
            Data Filtering and Sorting with Pandas
################################################################

# Filter passengers who survived
survivors = titanic_df[titanic_df['survived'] == 1]
# Sort survivors by age
sorted_df = survivors.sort_values('age')
# Sort survivors by class and age
sorted_df = survivors.sort_values(['pclass', 'age'], ascending=[False, True])
# Filter female passengers who survived
female_survivors = titanic_df[
    (titanic_df['survived'] == 1) & (titanic_df['sex'] == 'female')
]

female_survivors = titanic_df.query('survived == 1 & sex == "female"')


################################################################
            Data Visualization
################################################################


# Set display option to show all columns
pd.set_option('display.max_columns', None)
# Revert to Default Settings:
pd.reset_option('display.max_columns')
# Temporarily Change the Context:
with pd.option_context('display.max_columns', None):
    print(titanic_df.head())

# Print the first five entries
print(titanic_df.head())

# Print the last five entries
print(titanic_df.tail())

# Print the shape of the DataFrame
print(titanic_df.shape)
# Output: (891, 15)

# Print a concise summary of the DataFrame
titanic_df.info()
"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 15 columns):
 #   Column       Non-Null Count  Dtype
---  ------       --------------  -----
 0   survived     891 non-null    int64
 1   pclass       891 non-null    int64
 2   sex          891 non-null    object
 3   age          714 non-null    float64
 4   sibsp        891 non-null    int64
 5   parch        891 non-null    int64
 6   fare         891 non-null    float64
 7   embarked     889 non-null    object
 8   class        891 non-null    category
 9   who          891 non-null    object
 10  adult_male   891 non-null    bool
 11  deck         203 non-null    category
 12  embark_town  889 non-null    object
 13  alive        891 non-null    object
 14  alone        891 non-null    bool
dtypes: bool(2), category(2), float64(2), int64(4), object(5)
memory usage: 80.7+ KB
"""

# Print the descriptive statistics of the DataFrame
print(titanic_df.describe())
"""
         survived      pclass         age       sibsp       parch        fare
count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200
"""

# Key Parameters for describe()
# Parameter	            Description
# include='all'	        Includes all columns (numeric and non-numeric).
# include=[type]	    Specify data types to include (e.g., ['object', 'category']).
# exclude=[type]	    Exclude specific data types from the output.

# Generate descriptive statistics
titanic_stats = titanic.describe(include='all')

# Distribution of categorical data
print(titanic_df['sex'].value_counts())
"""
male      577
female    314
Name: sex, dtype: int64
"""

# Print the count of unique entries in 'embarked' column
# Returns the number of unique non-null values
print(titanic_df['embarked'].nunique()) # Output: 3

# Print the unique entries in 'embarked' column
# Returns an array of all unique values in the column, including NaN (if present).
print(titanic_df['embarked'].unique()) # Output: ['S' 'C' 'Q' nan]


# Calculate the numerical data range
age_range = titanic['age'].max() - titanic['age'].min()
print('Age Range:', age_range) # Age Range: 79.58

# Calculate the IQR
Q1 = titanic['age'].quantile(0.25)
Q3 = titanic['age'].quantile(0.75)
IQR = Q3 - Q1
print('Age IQR:', IQR) # Age IQR: 17.875


################################################################
                         Matplotlib
################################################################


import matplotlib.pyplot as plt

# Count total males and females
gender_data = titanic_df['sex'].value_counts()

# Create a bar chart
gender_data.plot(kind='bar', title='Sex Distribution')
plt.show()


# Types of Plots Supported by .plot()
# The .plot() method supports different kinds of plots by specifying the kind parameter:
#
# Plot Type	         kind Parameter	        Use Case
# Line Plot	         'line' (default)	    Visualizing trends in data.
# Bar Chart	         'bar'        	        Comparing categorical data (e.g., gender distribution).
# Horizontal Bar	 'barh'	                Similar to bar chart but horizontally oriented.
# Pie Chart	         'pie'	                Showing proportions of categories.
# Histogram	         'hist'	                Visualizing distributions of numeric data.

gender_data.plot(kind='bar', color='skyblue', alpha=0.7, grid=True)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Sex Distribution")
plt.show()


import seaborn as sns

# Set the seaborn default aesthetic parameters
sns.set(style="whitegrid")

################################################################
                         Seaborn
################################################################

# Set plot styling
sns.set(style="whitegrid", palette="Blues", font="Serif", font_scale=1.2)

# Create a plot
sns.countplot(x='pclass', data=titanic_df)
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='pclass', data=titanic_df, palette='coolwarm')
plt.title('Passenger Class Count')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Passenger Class')
plt.xticks(rotation=45)
plt.show()


################################################################
                         Histogram
################################################################

import seaborn as sns

titanic_df = sns.load_dataset('titanic')

# Increase the number of bins to 30 (default is 10)
sns.histplot(data=titanic_df, x='age', bins=30, kde=True)

# Give your plot a comprehensive title
plt.title('Age Distribution Among Titanic Passengers')

# Label your axes
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# A histogram using 'hue', 'multiple', and 'palette'
sns.histplot(data=titanic_df, x='age', hue="sex", multiple="stack", palette="pastel")

# A histogram using 'binwidth' and 'element'
sns.histplot(data=titanic_df, x="age", binwidth=1, element="step", color="purple")


################################################################
                         Barplot
################################################################

# Color-coded bar plot representing 'sex' and survival ('survived')
sns.countplot(x='sex', hue='survived', data=titanic_df, color="cyan",
              order=["female", "male"], orient='v').set_title('Sex and Survival Rates')
# hue - This parameter allows you to represent an additional categorical variable by colors. It becomes very handy in analyzing how the distribution of categories changes with respect to other categorical variables.
# color - This parameter lets you set a specific color for all the plot bars.
# order and hue_order - These parameters can be useful in arranging the bars in a specific order. You can provide an ordered list of categories to these parameters to adjust the ordering of bars.
# orient - This parameter can be used to change the plot's orientation. By default, it's set to 'v' for vertical plots. You can change it to 'h' for horizontal plots.



################################################################
              Scatter Plots and Correlation of Variables
################################################################

sns.scatterplot(x='age', y='fare', hue='pclass', style='sex', size='fare', sizes=(20, 200), data=titanic)
plt.title("Age vs Fare (Separate markers for Sex and Sizes for Fare)")
plt.show()

# style: This attribute will make different marks on the plot for different categories.
# size: This attribute can determine the size of a plotting mark using an additional variable. This represents another layer of information, providing you with a 3-dimensional plot.

# Correlation of all numeric variables in the Titanic dataset
corr_vals = titanic.corr(numeric_only=True)
print(corr_vals)


################################################################
                            Box Plots
################################################################


sns.boxplot(
    x='pclass', y='fare',
    hue='survived',
    data=titanic_df,
    palette='Set3', linewidth=1.5,
    order=[3,1,2], hue_order=[1,0],
    color='skyblue', saturation=0.7,
    dodge=True, fliersize=5
)
plt.title('Fares vs Passenger Classes Differentiated by Survival')
plt.show()


################################################################
               Heatmaps for Correlation Analysis
################################################################

# Calculate correlation matrix
correlation_matrix = titanic_df.corr(numeric_only=True)

# Create a heatmap
sns.heatmap(correlation_matrix, annot=True, cbar=True, vmin=-1, vmax=1)
# Show plot
plt.show()


# Building a color map
color_map = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(correlation_matrix, annot=True, cmap=color_map)

plt.show()


################################################################
              Missing Data
################################################################

# Detect missing values
missing_values = titanic_df.isnull()
print(missing_values.head(10))
"""
   survived  pclass    sex    age  ...   deck  embark_town  alive  alone
0     False   False  False  False  ...   True        False  False  False
1     False   False  False  False  ...  False        False  False  False
2     False   False  False  False  ...   True        False  False  False
3     False   False  False  False  ...  False        False  False  False
4     False   False  False  False  ...   True        False  False  False
5     False   False  False   True  ...   True        False  False  False
6     False   False  False  False  ...  False        False  False  False
7     False   False  False  False  ...   True        False  False  False
8     False   False  False  False  ...   True        False  False  False
9     False   False  False  False  ...   True        False  False  False

[10 rows x 15 columns]
"""

missing_values_count = titanic_df.isnull().sum()
print(missing_values_count)
"""
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
"""

-----  Dropping  -----------
# Copy the original dataset
titanic_df_copy = titanic_df.copy()

# Drop rows with missing values
titanic_df_copy.dropna(inplace=True)

# Check the dataframe
print(titanic_df_copy.isnull().sum())
# There will be no missing values in every column

# Detected missing values visualized
plt.figure(figsize=(10,6))
sns.heatmap(titanic_df.isnull(), cmap='viridis')
plt.show()

-----  Imputation  -----------
# Impute missing values using mean
 --- NOT RECOMMENDED  ---
titanic_df['age'].fillna(titanic_df['age'].mean(), inplace=True)

-----   USE THIS  ------
titanic_df['age'] = titanic_df['age'].fillna(titanic_df['age'].mean())

# Use a Dictionary with fillna (For Multiple Columns): If you're working with
# multiple columns, you can use the dictionary-based approach mentioned in the warning:
titanic_df.fillna({'age': titanic_df['age'].mean()}, inplace=True)

titanic_df['age'] = titanic_df['age'].apply(lambda x: titanic_df['age'].mean() if pd.isna(x) else x)


# Check the dataframe
print(titanic_df.isna().sum())
"""
survived         0
pclass           0
sex              0
age              0
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
"""

# Impute missing values using backward fill
--- NOT RECOMMENDED ---
titanic_df['age'].fillna(method='bfill', inplace=True)
-----   USE THIS  ------
titanic_df['age'] = titanic_df['age'].bfill()

# Check the dataframe
print(titanic_df.isnull().sum())
# The output is the same as in the previous example


################################################################
              Copy
################################################################

import copy
shallow_copy = copy.copy(original_df)


import copy
deep_copy = copy.deepcopy(original_df)

################################################################
             Encoding and Transforming Categorical Data
################################################################
-----  Label Encoding ----

--- NOT RECOMMENDED ---
# Label Encoding for 'sex'
titanic_df['sex_encoded'] = pd.factorize(titanic_df['sex'])[0]
print(titanic_df[['sex', 'sex_encoded']].head())
"""
      sex  sex_encoded
0    male            0
1  female            1
2  female            1
3  female            1
4    male            0
"""

--- best practices ---

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
titanic_df['sex_encoded'] = le.fit_transform(titanic_df['sex'])


# pd.factorize ensures every unique value is encoded as a number, even if NaN exists (which becomes -1).
# LabelEncoder does not handle NaN directly (raises an error).

# One-Hot Encoding for 'embark_town'
encoded_df = pd.get_dummies(titanic_df['embark_town'], prefix='town')
titanic_df = pd.concat([titanic_df, encoded_df], axis=1)
print(titanic_df.head())
"""
   survived  pclass     sex  ...  town_Cherbourg  town_Queenstown  town_Southampton
0         0       3    male  ...           False            False              True
1         1       1  female  ...            True            False             False
2         1       3  female  ...           False            False              True
3         1       1  female  ...           False            False              True
4         0       3    male  ...           False            False              True
"""

--- best practices ---
# To avoid multicollinearity issues in regression models, drop one of the dummy columns.
encoded_df = pd.get_dummies(titanic_df['embark_town'], prefix='town', drop_first=True)
titanic_df = pd.concat([titanic_df, encoded_df], axis=1)

# Scikit-learn’s OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')
encoded_array = ohe.fit_transform(titanic_df['embark_town'])
encoded_df = pd.DataFrame(encoded_array, columns=ohe.get_feature_names_out(['embark_town']))
titanic_df = pd.concat([titanic_df, encoded_df], axis=1)

################################################################
            Data Transformation and Scaling Techniques
################################################################
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset and drop rows with missing values
titanic_df = sns.load_dataset('titanic').dropna()

 -- best practice ---
titanic_df = sns.load_dataset('titanic').dropna(subset=['age', 'embarked'])

# Initialize the StandardScaler
std_scaler = StandardScaler()

# Fit and transform the 'age' column
titanic_df['age'] = std_scaler.fit_transform(np.array(titanic_df['age']).reshape(-1, 1))

# Check the transformed 'age' column
print(titanic_df['age'].head())
"""
1     0.152082
3    -0.039875
6     1.175852
10   -2.023430
11    1.431795
Name: age, dtype: float64
"""

from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
min_max_scaler = MinMaxScaler()
# Create a MinMaxScaler with feature range (-1, 1)
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))


# Fit and transform the 'fare' column
titanic_df['fare'] = min_max_scaler.fit_transform(np.array(titanic_df['fare']).reshape(-1, 1))

---------------------------------------------
# Import the necessary libraries
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Load the Titanic dataset
titanic_df = sns.load_dataset('titanic').dropna()

# Initialize the scalers
std_scaler = StandardScaler()
min_max_scaler = MinMaxScaler()
robust_scaler = RobustScaler()

# Scale the 'age' column of the dataset using Standard Scaler
titanic_df['age_std'] = std_scaler.fit_transform(np.array(titanic_df['age']).reshape(-1, 1))

# Scale the 'fare' column of the dataset using the Min-Max Scaler
titanic_df['fare_minmax'] = min_max_scaler.fit_transform(np.array(titanic_df['fare']).reshape(-1, 1))

# Scale the 'fare' column of the dataset using Robust Scaler
titanic_df['fare_robust'] = robust_scaler.fit_transform(np.array(titanic_df['fare']).reshape(-1, 1))

# Print the first 5 rows of the modified dataset
print(titanic_df.head())

# Drop the original 'fare' column
titanic_df.drop(columns=['fare'], inplace=True)


################################################################
            Outlier Detection
################################################################
--- Z-score Method ---

# Calculate Z-scores
titanic_df['age_zscore'] = np.abs((titanic_df.age - titanic_df.age.mean()) / titanic_df.age.std())

# Get rows of outliers according to the Z-score method (using a threshold of 3)
outliers_zscore = titanic_df[(titanic_df['age_zscore'] > 3)]
print(outliers_zscore)
"""
     survived  pclass   sex   age  ...  embark_town  alive  alone age_zscore
630         1       1  male  80.0  ...  Southampton    yes   True   3.462699
851         0       3  male  74.0  ...  Southampton     no   True   3.049660

[2 rows x 16 columns]
"""
