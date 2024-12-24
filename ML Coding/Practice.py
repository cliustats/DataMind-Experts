1. When to Refresh an ML Model? Why model performance might differ in production vs in development?
      https://www.youtube.com/watch?v=w9ipP9kPpFc&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=22
2. ML Training Data vs. Testing Data  https://www.youtube.com/watch?v=_Y7E2YKfuFM&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=20
3. Handling Exploding Gradients in Machine Learning   https://www.youtube.com/watch?v=Y5DVYLmmdrw&list=PLrtCHHeadkHqYX7O5cjHeWHzH2jzQqWg5&index=19
  1) Clipping the gradidents
  2) batch normalization
  3) Change the architecture. reduce the hidden layers, or transformer with skip connections




# Debug ----- pdb
# The cd command is used to change the current working directory
$ cd # cd .. back up one ditectory
$ ls -l # show the size
$ ls -a # list all files, including hidden ones, in a directory
$ pwd # print working directory
$ mkdir # create new directories
$ mkdir -p projects/first_project # create a directory with sub-directories
$ rmdir projects # remove empty directories
$ rm -rf projects # remove non-empty directories
$ mv ~/Downloads/pic.png ~/Documents/Photo/pic.png # move
$ cp ~/Downloads/pic.png ~/Documents/Photography  # copy
$ cat #(concatenate), read data from a specified file and print the output.
$ less #terminal pager that outputs the content of the specified file one screen at a time.
$ find / -name 'my_file.txt' # find.  / home directory

ls -l | grep ".py"
# Lists all files with detailed info (ls -l) and filters
# only files containing .py in their names.
&&
#purpose of the && operator in the shell.
#Runs the second command only if the first command succeeds.
echo $((5 + 3))
# Outputs the result of the arithmetic operation: 8
cat file.txt | sort | uniq -c | sort -nr
# Reads file.txt, sorts the lines, counts unique occurrences,
# and sorts by count in descending order.
chmod: Changes file permissions.
chown: Changes file ownership.
du -sh <directory_name> # disk usage of a directory
du -h data.csv # check the size of a file
ps aux # Displays detailed information about all running processes.

################################################################
                        Virtual Environment
################################################################

#!/bin/bash

# Isolates dependencies for each project.
# Prevents conflicts between package versions.
# Makes projects more portable and reproducible.

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


##########################
# Reading a file
with open('example.txt', 'r') as file:
    content = file.read()
    print(content)

with open('example.txt', 'w') as file:
    file.write('Hello, World!\n')
    file.write('This is a new line.')

# Appending to a file
with open('example.txt', 'a') as file:
    file.write('\nThis line is appended.')

# Reading a file line by line
with open('example.txt', 'r') as file:
    for line in file:
        print(line.strip())  # Removes extra newlines

# Reading a File Safely (Catching Exceptions)
try:
    with open('example.txt', 'r') as file:
        content = file.read()
        print(content)
except FileNotFoundError:
    print('The file does not exist!')


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
#                          Unit test
################################################################
# -- vector
#    --- vector.py    # main python file
#    --- __init__.py  # make the folder to a package
# -- tests
#    --- test_vector.py
#    --- __init__.py

# vector.py
class Vector:
    def __init__(self, x, y):
        if isintance(x, (int, float)) and \
           isintance(y, (int, float)):
            self.x = x
            self.y = y
        else:
            raise ValueError("Not a number")

    def add(self, other):
        return Vector(self.x + other.x,
                      self.y + other.y)


# test_vector.py
import sys
import unittest
from vector import Vector

class TestVector(unittest.TestCase):
    # def setUp(self):
    #     print("start")
    @classmethod
    def setUpClass(cls):
        print("start")

    def tearDown(self):
        print("End")

    def test_init(self):   # have to start with test_
        v = Vector(1, 2)
        self.assertEqual(v.x, 1)
        self.assertEqual(v.y, 1)

        with self.assertRaises(ValueError):
            v = Vector("1", "2")


    @unittest.skipIf(sys.platform == 'win32', "Do not support Windows")
    def test_add(self):
        v1 = Vector(1, 2)
        v2 = Vector(2, 3)
        v3 = v1.add(v2)
        self.assertEqual(v3.x, 3)


# shell: python3 -m unittest
# shell: python3 -m unittest tests.test_vector.TestVector.test_add


# assertEqual(a, b) # when fails, gives more info
# assertNotEqual(a, b)
# assertTrue(x)     # when fails, True != False
# assertFalse(x)
# assertIs(a, b)
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

--- The IQR Method ---
# Calculate IQR
Q1 = titanic_df['age'].quantile(0.25)
Q3 = titanic_df['age'].quantile(0.75)
IQR = Q3 - Q1

# Define Bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Get rows of outliers according to IQR method
outliers_iqr = titanic_df[(titanic_df['age'] < lower_bound) | (titanic_df['age'] > upper_bound)]
print(outliers_iqr)
"""
     survived  pclass   sex   age  ...  embark_town  alive  alone age_zscore
33          0       2  male  66.0  ...  Southampton     no   True   2.498943
54          0       1  male  65.0  ...    Cherbourg     no  False   2.430103
96          0       1  male  71.0  ...    Cherbourg     no   True   2.843141
116         0       3  male  70.5  ...   Queenstown     no   True   2.808721
280         0       3  male  65.0  ...   Queenstown     no   True   2.430103
456         0       1  male  65.0  ...  Southampton     no   True   2.430103
493         0       1  male  71.0  ...    Cherbourg     no   True   2.843141
630         1       1  male  80.0  ...  Southampton    yes   True   3.462699
672         0       2  male  70.0  ...  Southampton     no   True   2.774301
745         0       1  male  70.0  ...  Southampton     no  False   2.774301
851         0       3  male  74.0  ...  Southampton     no   True   3.049660

[11 rows x 16 columns]
"""
--- Dropping them ---
# Using the Z-score method
titanic_df = titanic_df[titanic_df['age_zscore'] <= 3]

# Using the IQR method
titanic_df = titanic_df[(titanic_df['age'] >= lower_bound) & (titanic_df['age'] <= upper_bound)]

--- Replacing them with another value (mean, median, mode, etc.) ---
# using mean
titanic_df.loc[titanic_df['age_zscore'] > 3, 'age'] = titanic_df['age'].mean()

# using median
titanic_df.loc[(titanic_df['age'] < lower_bound) | (titanic_df['age'] > upper_bound), 'age'] = titanic_df['age'].median()

# DataFrame.loc[row_indexer, column_indexer]
# row_indexer: Filters rows based on labels, conditions, or slices.
# column_indexer: Specifies the columns to select or modify.


################################################################
            Correlated Features
################################################################

# Calculate the correlation matrix
corr_matrix = titanic_df.corr(numeric_only=True)

# Let's visualize this correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)

# Add a title
plt.title('Heatmap of the Correlation Matrix')

# Show the plot
plt.show()

# Now, let's remove a redundant feature
# Choose age and parch, as they are highly correlated
clean_df = titanic_df.drop('age', axis=1)

# Print the first 5 rows of the cleaned dataframe
print(clean_df.head())

################################################################
            Feature Engineering
################################################################

# Create a new feature, 'family_size'
titanic_df['family_size'] = titanic_df['sibsp'] + titanic_df['parch'] + 1

# Define the bin edges
age_bins = [0, 12, 18, 30, 45, 100]

# Define the bin labels
age_labels = ['Child', 'Teenager', 'Young Adult', 'Middle Age', 'Senior']

# Create the age group feature
titanic_df['age_group'] = pd.cut(titanic_df['age'], bins=age_bins, labels=age_labels)

# Show the first few rows of the data
print(titanic_df.head())
"""
   survived  pclass     sex   age  ...  alive  alone  family_size    age_group
0         0       3    male  22.0  ...     no  False            2  Young Adult
1         1       1  female  38.0  ...    yes  False            2   Middle Age
2         1       3  female  26.0  ...    yes   True            1  Young Adult
3         1       1  female  35.0  ...    yes  False            2   Middle Age
4         0       3    male  35.0  ...     no   True            1   Middle Age

[5 rows x 17 columns]
"""
------
# pd.cut() doesn't handle missing values
# By default, pd.cut() includes the left edge and excludes the right edge ([left, right)). To change this behavior, use the right=False parameter:

titanic["fare_per_age"] = titanic["fare"] / titanic["age"]
titanic["fare_per_age"] = titanic["fare_per_age"].replace([np.nan, np.inf, -np.inf], 0)

################################################################
            Matrix Operations in Numpy
################################################################

F = np.linalg.inv(E)  # Finding the inverse of matrix E
FP = np.linalg.pinv(A)  # Finding the pseudo-inverse matrix of A

# np.linalg.inv(a) can only be used for square matrices that are invertible,
# while np.linalg.pinv(a) can be used for any matrix.
# If the original matrix is singular or non-square, np.linalg.inv(a) will result in an error,
#  whereas np.linalg.pinv(a) will still return a result.

G = np.transpose(A)

# Assuming these are two features from our dataset
feature_1 = np.array([[123], [456], [789]])
feature_2 = np.array([[321], [654], [987]])

# Combine the two features into one matrix
data_features = np.hstack((feature_1, feature_2))
print(data_features)
"""
[[123 321]
 [456 654]
 [789 987]]
"""

# np.hstack():
#
# Stacks arrays horizontally (column-wise).
# Requires that the arrays have the same number of rows.

normalized_data_features = data_features / np.linalg.norm(data_features)

row_norms = np.linalg.norm(data_features, axis=1, keepdims=True)
row_normalized = data_features / row_norms

normalized_data_features_minmax = (data_features - np.min(data_features)) / (np.max(data_features) - np.min(data_features))

----------------
axis=0: Operates down the rows (column-wise).
axis=1: Operates across the columns (row-wise).


Operation	            Axis	   Effect
df.sum(axis=0)       	axis=0	   Column-wise sum (sums all rows for each column).
df.sum(axis=1)	        axis=1	   Row-wise sum (sums all columns for each row).
df.drop(axis=0)	        axis=0	   Drops rows.
df.drop(axis=1)	        axis=1	   Drops columns.
df.apply(func, axis=0)	axis=0	   Applies func to each column.
df.apply(func, axis=1)	axis=1	   Applies func to each row.
np.sum(array, axis=0)	axis=0	   Sum along rows (column-wise operation).
np.sum(array, axis=1)	axis=1	   Sum along columns (row-wise operation).


################################################################
             Advanced Functions in Pandas
################################################################

# Create a simple dataframe
data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
       'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
       'Sales': [200, 120, 340, 124, 243, 350]}
df = pd.DataFrame(data)

# for key in df_grouped.groups:
#     print(f"Group Key: {key}")

# Apply groupby
df_grouped = df.groupby('Company')
for key, item in df_grouped:
    print("\nGroup Key: {}".format(key))
    print(df_grouped.get_group(key), "\n")
"""
Group Key: FB
  Company Person  Sales
4      FB   Carl    243
5      FB  Sarah    350


Group Key: GOOG
  Company   Person  Sales
0    GOOG      Sam    200
1    GOOG  Charlie    120


Group Key: MSFT
  Company   Person  Sales
2    MSFT      Amy    340
3    MSFT  Vanessa    124
"""

grouped = df.groupby('Company')
print(grouped.sum())
"""
             Person  Sales
Company
FB        CarlSarah    593
GOOG     SamCharlie    320
MSFT     AmyVanessa    464
"""

# Create a dataframe
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})

# Define a function
def get_sum(row):
    return row.sum()

# Apply the function
df['sum'] = df[['C', 'D']].apply(get_sum, axis=1)

print(df)
"""
     A      B         C         D       sum
0  foo    one -0.343200  0.184665 -0.158535
1  bar    one  0.058870  1.835614  1.894484
2  foo    two  0.801743 -0.184409  0.617333
3  bar  three  0.935406  0.124109  1.059515
4  foo    two  0.782074  0.583470  1.365544
5  bar    two  0.138934  0.710407  0.849341
6  foo    one  0.364633  1.147963  1.512596
7  foo  three -1.364677  1.719538  0.354861
"""
----------------------------------------------------
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Fetch the dataset
data = fetch_california_housing(as_frame=True)

# create a DataFrame
housing_df = pd.DataFrame(data=data.data, columns=data.feature_names)

# Define income category
housing_df['income_cat'] = pd.cut(housing_df['MedInc'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

# Group by income category and calculate the average population
average_population = housing_df.groupby('income_cat').apply(lambda x: x['Population'].mean())

print(average_population)
"""
income_cat
1    1105.806569
2    1418.232336
3    1448.062465
4    1488.974718
5    1389.890347
dtype: float64
"""

################################################################
             Optimization
################################################################

# The first technique is choosing the pd.Categorical data type (or use .astype('category')) specifically for categorical data (data that takes on a limited, usually fixed, number of possible values), which can yield significant savings in memory.

df['Type'] = pd.Categorical(df['Type'])
df['MedInc'] = df['MedInc'].astype('category')

# Downcast data type for 'AveBedrms' column
df['AveBedrms'] = pd.to_numeric(df['AveBedrms'], downcast='float')
df['Population'] = df['Population'].astype('int32')

# Regular way
df_copy = df[df['Population'] > 1000]
df_copy.dropna(inplace=True)

# Optimized way
df[df['Population'] > 1000].dropna(inplace=True)

------------------ Change Data Type ------

# Load the California Housing dataset
california = datasets.fetch_california_housing()
df = pd.DataFrame(data=np.c_[california['data'], california['target']], columns=california['feature_names'] + ['target'])

def memory_usage_pandas(df):
    bytes = df.memory_usage(deep=True).sum()
    return bytes / 1024**2  # Convert bytes to megabytes

original_memory = memory_usage_pandas(df)

# Optimize memory usage in Pandas using categorical data types
# California Housing dataset does not have any Categorical features, so we will use downcasting
df['AveBedrms'] = pd.to_numeric(df['AveBedrms'], downcast='float')
df['AveRooms'] = pd.to_numeric(df['AveRooms'], downcast='float')
optimized_memory = memory_usage_pandas(df)

print(f'Original memory usage: {original_memory} MB')
print(f'Optimized memory usage: {optimized_memory} MB')
print(f'Memory saved: {original_memory - optimized_memory} MB')

percentage_saved = (start_mem - end_memory) / start_mem * 100
print(f"The percentage of the memory saved is {percentage_saved:.2f}%")

----------------------------------
import numpy as np
import pandas as pd
from sklearn import datasets
import time

# Load the California Housing Dataset
california = datasets.fetch_california_housing()
df = pd.DataFrame(
    data=np.c_[california['data'], california['target']],
    columns=california['feature_names'] + ['target']
)
print("Initial DataFrame:")
print(df.head())

# Function to calculate memory usage
def get_memory_usage(df):
    bytes = df.memory_usage(deep=True).sum()
    mb = bytes / 1024**2  # Convert bytes to megabytes
    return mb

# Grab initial memory usage
original_memory = get_memory_usage(df)

# Optimize the 'MedInc' feature
start_time = time.time()

# Log transformation on 'MedInc'
df['MedInc'] = df['MedInc'].apply(lambda x: np.log(x))

# Cut 'MedInc' into 5 bins and label them 0-4
df['MedInc'] = pd.cut(df['MedInc'], bins=5, labels=[0, 1, 2, 3, 4])

# Convert 'MedInc' to categorical type
df['MedInc'] = df['MedInc'].astype('category')

end_time = time.time()

# Compute the memory usage after tuning
optimized_memory = get_memory_usage(df)

# Compute the memory reduction percentage
percentage_saved = (original_memory - optimized_memory) / original_memory * 100

# Print the results
print(f"Original memory usage: {original_memory:.2f} MB")
print(f"Optimized memory usage: {optimized_memory:.2f} MB")
print(f"Memory saved: {percentage_saved:.2f}%")
print(f"Time taken for optimization: {end_time - start_time:.4f} seconds")

-------------------------------------
data = {
    "Gene": ["Gene A", "Gene B", "Gene C", "Gene D"],
    "Sequence": ["ATCGTACGA", "CGATCGATG", "TAGCTAG", "CGTAGCTA"]
}

df_gene = pd.DataFrame(data)

df_gene['Length'] = df_gene['Sequence'].apply(len)
df_gene['A_Count'] = df_gene['Sequence'].apply(x: x.count('A'))
df_gene['Reversed'] = df_gene['Sequence'].apply(x: x[::-1])
df_gene['RNA'] = df_gene['Sequence'].apply(lambda x: x.replace('T', 'U'))  # replace T with U
# To compute the complementary sequence (replace bases with their complements: A ↔ T, C ↔ G):
complement_map = str.maketrans({"A": "T", "T": "A", "C": "G", "G": "C"})
df_genes["Complement"] = df_genes["Sequence"].apply(lambda x: x.translate(complement_map))
print(df_genes)
