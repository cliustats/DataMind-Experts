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
#                        Collection
################################################################
# Collection = single 'variable' used to store multiple values

# List  = [] ordered and changable. duplicates ok
# Set   = {} unordered and immutable, but Add/Remove ok. NO DUPLICATES
# tuple = () ordered and unchangable. duplicates ok, faster

# LIST
fruits = ['apple', 'banana', 'orange']
print(dir(fruits))
print(help(fruits))
fruits[0] = 'pineapple'
fruits.append()
fruits.remove()
fruits.insert(0, 'pineapple')
fruits.sort()
fruits.reverse()
fruits.clear()
fruits.index('apple')
fruits.count('banana')


def safe_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return -1  # or any default value

fruits = ['banana', 'apple', 'cherry']
index = safe_index(fruits, 'orange')
print(index)  # Output: -1

# SET
fruits = {'apple', 'banana', 'orange'}
fruits.add('pineapple')
fruits.remove('apple')
fruits.pop()  # randomly delete an element
fruits.clear()

# TUPLE
fruits = ('apple', 'banana', 'orange')
fruits.index('apple')
fruits.count('pineapple')

# dictionary =  a collection of {key:value} pairs
#                       ordered and changeable. No duplicates

capitals = {"USA": "Washington D.C.",
                    "India": "New Delhi",
                    "China": "Beijing",
                    "Russia": "Moscow"}

print(dir(capitals))
print(help(capitals))
print(capitals.get("Japan"))

if capitals.get("Russia"):
   print("That capital exists")
else:
   print("That capital doesn't exist")

capitals.update({"Germany": "Berlin"})
capitals.update({"USA": "Detroit"})
capitals.pop("China")
capitals.popitem()  # pop the lastest key-value pair that was inserted
capitals.clear()

keys = capitals.keys()
for key in capitals.keys():
  print(key)

values = capitals.values()
for value in capitals.values():
print(value)

items = capitals.items()
for key, value in capitals.items():
   print(f"{key}: {value}")

######################

# *args       = allows you to pass multiple non-key arguments
# **kwargs    = allows you to pass multiple keyword-arguments
#             * unpacking operator

def shipping_label(*args, **kwargs):
    for arg in args:
        print(arg, end=" ")
    print()

    if "apt" in kwargs:
        print(f"{kwargs.get('street')} {kwargs.get('apt')}")
    elif "pobox" in kwargs:
        print(f"{kwargs.get('street')}")
        print(f"{kwargs.get('pobox')}")
    else:
        print(f"{kwargs.get('street')}")

    print(f"{kwargs.get('city')}, {kwargs.get('state')} {kwargs.get('zip')}")

shipping_label("Dr.", "Spongebob", "Squarepants",
               street="123 Fake St.",
               pobox="PO box #1001",
               city="Detroit",
               state="MI",
               zip="54321")


######################
# Exercise: Don't modify the list when you are iterating it
items = ['A', 'B', 'C', 'D', 'E']

for item in items:
    if item == 'B':
        items.remove('B')
    else:
        print(item)

# Output: ['A', 'D', 'E']

# Correct way
new_items = []

for item in items:
    if item == 'B':
        continue
    else:
        print(item)
        new_items.append(item)
print(new_items)
# Output: ['A', 'C', 'D', 'E']


######################
# Module = a file containing code you want to include in your program
#          use 'import' to include a module (built-in or your own)
#          useful to break up a large program reusable seperate files

######################
# Scope resolution = (LEGB) local -> enclosed -> global -> built-in


######################
import random

low = 1
high = 20
options = ('Rock', 'Paper', ‘Scissors’)
cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

number = random.random  # return floating number between 0 and 1
number = random.randint(1, 6)
number = random.randint(low, high)
choice = random.choice(options)
random.shuffle(cards)

######################
# Order in Function Calls:
#
# Positional arguments → Default arguments → *args → Keyword arguments → **kwargs.
######################
for number in range(1, 11):
    print(number, end=" ")

print("1", "2", "3", "4", "5", sep="-")

######################
# Iterables = An object/collection that can return its elements one at a time,
#             allowing it to be iterated over in a loop
my_list = [1, 2, 3, 4, 5]
my_tuple = (1, 2, 3, 4, 5)
my_set = {"apple", "orange", "banana", "coconut"}
my_name = "Bro Code"
my_dictionary = {'A': 1, 'B': 2, 'C': 3}

######################
# Membership operators = used to test whether a value or variable is found in a sequence
#                       (string, list, tuple, set, or dictionary)
#                       1. in
#                       2. not in

######################
# List comprehension = A concise way to create lists in Python
#                      Compact and easier to read than traditional loops
#                      [expression for value in iterable if condition]

######################
# if _name_ == __main__: (this script can be imported OR run standalone)
#                        Functions and classes in this module can be reused
#                        without the main block of code executing
#
# Good practice (code is modular, helps readability, leaves no global variables, avoids unintended execution)
#
# Ex. library = Import library for functionality.  When running library directly, display a help page.


def main():
    pass

if __name__ == '__main__':
    main()

################################################################
#                   Object Oriented Programming
################################################################

# object = A "bundle" of related attributes (variables) and methods (functions)
# Ex. phone, cup, book
# You need a "class" to create many objects
#
# class  = (blueprint) used to design the structure and layout of an object

class Car:
    def __init__(self, model, year, color, for_sale):   # constructer method. dunder method
        self.model = model   # RHS: Parameter, assign it to the object
        self.year = year
        self.color = color
        self.for_sale = for_sale

    def drive(self):
        print(f"You drive the {self.color} {self.model}")

    def stop(self):
        print(f"You stop the {self.color} {self.model}")

    def describe(self):
        print(f"{self.year} {self.color} {self.model}")


car1 = Car('Mustang', 2024, 'red', False)
print(car1.model)  # .: attribute access operator

##################################################################
# class variables = Shared among all instances of a class
#                   Defined outside the constructor
#                   Allow you to share data among all objects created from that class

class Student:
    class_year = 2024
    num_students = 0

    def __init__(self, name, age):
        self.name = name
        self.age = age
        Student.num_students += 1


student1 = Student('Alice', 25)
print(student1.name)
print(student1.age)
print(student1.class_year) # Bad practice
print(Student.class_year)  # Good practice

print(f"My graduating class of {Student.class_year} has {Student.num_students} students")


##################################################################
# Inheritance = Inherit attributes and methods from another class
#               Helps with code reusability and extensibility
#               class Child(Parent)
#               class Sup(Super)

class Animal:
    def __init__(self, name):
        self.name = name
        self.is_alive = True

    def eat(self):
        print(f"{self.name} is eating")

    def sleep(self):
        print(f"{self.name} is sleeping")


class Dog(Animal):
    def speak(self):
        print("Woof")


class Cat(Animal):
    def speak(self):
        print("Meow")


class Mouse(Animal):
    def speak(self):
        print("Squeek")


dog = Dog("Scooby")
cat = Cat("Garfield")
mouse = Mouse("Micky")


##################################################################
# multiple inheritance = inherit from more than one parent class
#                        C(A, B)
# multilevel inheritance = inherit from a parent which inherit from another parent
#                        C(B) <- B(A) <- A

class Animal:
    def __init__(self, name):
        self.name = name

    def eat(self):
        print(f"{self.name} is eating")

    def sleep(self):
        print(f"{self.name} is sleeping")


class Prey(Animal):
    def flee(self):
        print(f"{self.name} is fleeing")


class Predator(Animal):
    def hunt(self):
        print(f"{self.name} is hunting")


class Rabbit(Prey):
    pass


class Hawk(Predator):
    pass


class Fish(Prey, Predator):
    pass


rabbit = Rabbit("Bugs")
hawk = Hawk("Tony")
fish = Fish("Nemo")

rabbit.hunt()


##################################################################
# Abstract class: A class that cannot be instantiated on its own; Meant to be subclassed.
#                 They can contain abstract methods, which are declared but have no implementation.
#                 Abstract classes benefits:
#                 1. Prevents instantiation of the class itself
#                 2. Requires children to use inherited abstract methods
# Abstract base classes define a blueprint for subclasses, requiring them to implement specific methods.
# Any class with at least one @abstractmethod cannot be instantiated directly.


from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def go(self):
        pass

    @abstracmethod
    def stop(self):
        pass


class Car(Vehicle):
    def go(self):
        print("You drive the car")

    def stop(self):
        print("You stop the car")


class Motorcycle(Vehicle):
    def go(self):
        print("You ride the motorcycle")

    def stop(self):
        print("You stop the motorcycle")


class Boat(Vehicle):
    def go(self):
        print("You sail the boat")

    def stop(self):
        print("You anchor the boat")

##################################################################
# super() = Function used in a child class to call methods from a parent class (superclass)
#           Allows you to extend the functionality of the inherited methods

class Shape:
    def __init__(self, color, is_filled):
        self.color = color
        self.is_filled = is_filled

    def describe(self):
        print(f"It is {self.color} and {'filled' if self.is_filled else 'not filled'}")


class Circle(Shape):
    def __init__(self, color, is_filled, radius):
        super().__init__(color, is_filled)
        self.radius = radius

    def describe(self):
        super().describe()
        print(f"It is a circle with an area of {3.14 * self.radius * self.radius}cm^2")


class Square:
    def __init__(self, color, is_filled, width):
        super().__init__(color, is_filled)
        self.width = width

    def describe(self):
        print(f"It is a square with an area of {self.width * self.width}cm^2")
        super().describe()


class Triangle:
    def __init__(self, color, is_filled, width, height):
        super().__init__(color, is_filled)
        self.width = width
        self.height = height

    def describe(self):
        print(f"It is a triangle with an area of {self.width * self.height / 2}cm^2")
        super().describe()


##################################################################
# Polymorphism = Greek word that means to "have many forms or faces"
#                Poly = Many
#                Morphe = Form
#
# TWO WAYS TO ACHIEVE POLYMORPHISM
# 1. Inheritance = An object could be treated of the same type as a parent class
# 2. "Duck typing" = Object must have necessary attributes/methods

from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass


class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14 * self.radius**2


class Square(Shape):
    def __init__(self, side):
        self.side = side

    def area(self):
        return self.side**2


class Triangle(Shape):
    def __init__(self, base, height):
        self.base = base
        self.height = height

    def area(self):
        return self.base * self.height * 0.5


class Pizza(Circle):
    def __init__(self, topping, radius):
        super().__init__(radius)
        self.topping = topping


shapes = [Circle(4), Square(5), Triangle(6, 7), Pizza("pepperoni", 15)]

for shape in shapes:
    print(f"{shape.area()}cm²")

##################################################################
# "Duck typing" = Another way to achieve polymorphism besides Inheritance
#                 Object must have the minimum necessary attributes/methods
#                 "If it looks like a duck and quacks like a duck, it must be a duck."


##################################################################
# Aggregation = Represents a relationship where one object (the whole)
#               contains references to one or more INDEPENDENT objects (the parts)

class Library:
    def __init__(self, name):
        self.name = name
        self.books = []

    def add_book(self, book):
        self.books.append(book)

    def list_books(self):
        return [f"{book.title} by {book.author}" for book in self.books]

class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

library = Library("New York Public Library")

book1 = Book("Harry Potter...", "J.K. Rowling")
book2 = Book("The Hobbit", "J. R. R. Tolkein")
book3 = Book("The Colour of Magic", "Terry Pratchett")

library.add_book(book1)
library.add_book(book2)
library.add_book(book3)

print(library.name)

for book in library.list_books():
    print(book)

##################################################################
# Aggregation = A relationship where one object contains references to other INDEPENDENT objects
# "has-a" relationship
#
# Composition = The composed object directly owns its components, which cannot exist independently
# "owns-a" relationship

class Engine:
    def __init__(self, horse_power):
        self.horse_power = horse_power


class Wheel:
    def __init__(self, size):
        self.size = size


class Car:
    def __init__(self, make, model, horse_power, wheel_size):
        self.make = make
        self.model = model
        self.engine = Engine(horse_power)
        self.wheels = [Wheel(wheel_size) for _ in range(4)]

    def display_car(self):
        return f"{self.make} {self.model} {self.engine.horse_power}(hp) {self.wheels[0].size}in"


##################################################################
# Nested class = A class defined within another class
#                 class Outer:
#                      class Inner:
#
# Benefits: Allows you to logically group classes that are closely related
#           Encapsulates private details that aren't relevant outside of the outer class
#           Keeps the namespace clean; reduces the possibility of naming conflicts

class Company:
    class Employee:
        def __init__(self, name, position):
            self.name = name
            self.position = position

        def get_details(self):
            return f"{self.name} {self.position}"

    def __init__(self, company_name):
        self.company_name = company_name
        self.employees = []

    def add_employee(self, name, position):
        new_employee = self.Employee(name, position)
        self.employees.append(new_employee)

    def list_employees(self):
        return [employee.get_details() for employee in self.employees]


company1 = Company("Krusty Krab")
company2 = Company("Chum Bucket")

company1.add_employee("Eugene", "Manager")
company1.add_employee("Spongebob", "Cook")
company1.add_employee("Squidward", "Cashier")

company2.add_employee("Sheldon", "Manager")
company2.add_employee("Karen", "Assistant")

for employee in company2.list_employees():
    print(employee)

################################################################
#                        Decorator
################################################################







################################################################
#                        DS Project
################################################################

# https://github.com/everyday-data-science/Data_Science_Projects/blob/main/Sony%20Research/Data/Sony_Research.ipynb


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
    list(outliers_iqr.items()), columns=['Column', 'Number of Outliers']
)
outlier_df

class_balance = df['churn'].value_counts(normalize=True) * 100
print("Class Balance in dataframe (as percentile):")
print(class_balance)

plt.figure(figsize=(6,4))
sns.countplot(x='churn', data=df, palette='viridis')
plt.title('Class Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()


df['area code'] = df['area code'].astype('object')

df['phone number'].str.replace('-', '').str.len().value_counts()

df = df.drop('phone number', axis=1)

numeric_features = df.select_dtypes(include=['float64', 'int64']).columns

pearson_correlation_matrix = df[numeric_features].corr(method='pearson')

plt.figure(figsize=(10, 6))
sns.headmap(pearson_correlation_matrix, annot=True, fmt='.2f', cmap='Dark2')
plt.title('Pearson Correlation')
plt.show()


from sklearn.model_selection import train_test_split


X = df.drop(columns=['churn'])
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"The shape of the training set is {X_train.shape}")

df.select_dtypes(include=['category', 'object']).columns

# Encoding by direct mapping
y_train = y_train.map({True: 1, Flase: 0})

# Frquncy encoding
grouped_data = pd.concat([X_train['area code'], y_train], axis=1).groupby('area code').agg(
    churn_rate=('churn', 'mean'),
    total_customers=('churn', 'count')
)

churn_rate = merged_df.groupby('area code').mean().rename(columns={'churn': 'churn_rate'})

# Target Encoding
pd.concat([X_train["state"], y_train], axis=1).groupby("state").agg(
    churn_rate=("churn", "mean")
).sort_values(by="churn_rate", ascending=False)


# Step 1: Calculate frequency encoding based on the training set
target_encoding_state = (
    pd.concat([X_train["state"], y_train], axis=1)
    .groupby("state")
    .agg(churn_rate=("churn", "mean"))
)

# Step 2: Map the frequencies back to the training and test sets
X_train["state"] = X_train["state"].map(target_encoding_state["churn_rate"])
X_test["state"] = X_test["state"].map(target_encoding_state["churn_rate"])
X_train.head()



# State is a categorical column with high cardinality: 51 Unique values
# Extra Info: When you have high cardinality, some other things you can try are:

# Frequency Encoding: This method assigns the frequency of each category as the encoded value. It helps capture the representation of each state without increasing dimensionality.

# Hashing Encoding: This is useful for very high-cardinality data, as it hashes the categories into a fixed number of buckets.

# One-Hot Encoding with Thresholding: For high cardinality, One-Hot Encoding could be expensive (increasing feature dimensionality). However, you can limit One-Hot Encoding to the most frequent states (e.g., top n states), and group the remaining states into an "Other" category.


exclude_columns = ['state', 'area code', 'international plan', 'voice mail plan',
       'binned_voicemail', 'binned_customer_service_calls', 'total_day_charge_per_minute',
                   'total_eve_charge_per_minute', 'total_night_charge_per_minute',
                   'total_intl_charge_per_minute']
columns_to_scale = X_train.columns.difference(exclude_columns)

# Apply Robust Scaler only on the specified columns
scaler = RobustScaler()
X_train[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
X_test[columns_to_scale] = scaler.transform(X_test[columns_to_scale])
X_train.head()




from sklearn.ensemble import RandomForestClassifier
# Create and fit the Random Forest model
rf_model = RandomForestClassifier(n_estimators=500)
rf_model.fit(X_train, y_train)

# Get feature importance
importance = rf_model.feature_importances_

# Create a DataFrame for feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title('Feature Importance in Churn Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()


from sklearn.linear_model import LinearRegression




# Save the Best Model
joblib.dump(best_lgb_model, 'best_lgb_model.pkl')



encoding_state = target_encoding_state.to_dict()['churn_rate']
encoding_area_code = freq_encoding_area_code.to_dict()
encoding_binned_csc = target_encoding_binned_csc.to_dict()['churn_rate']

def encode(X):

  # Map 'international plan' and 'voice mail plan'
    X['international plan'] = X['international plan'].map({'no': 0, 'yes': 1})
    X['voice mail plan'] = X['voice mail plan'].map({'no': 0, 'yes': 1})

    # Target encoding for 'state'
    X['state'] = X['state'].map(encoding_state)

    # Frequency encoding for 'area code'
    X['area code'] = X['area code'].map(encoding_area_code)

    X['binned_customer_service_calls'] = pd.cut(X['customer service calls'], bins=[-1, 3, 5, np.inf], labels=['Low', 'Medium', 'High'])
    X['binned_customer_service_calls'] = X['binned_customer_service_calls'].map(encoding_binned_csc)

    return X

def log_transform(X):
    # Apply log transformation to 'total intl calls'
    X['total_intl_calls_log'] = np.log1p(X['total intl calls'])

    return X

columns_to_drop = ['phone number', 'number vmail messages', 'total day minutes',
                    'total eve minutes', 'total night minutes', 'total intl minutes',
                   'customer service calls', 'total intl calls']

columns_to_scale = ['account length', 'total day calls', 'total day charge',
                    'total eve calls', 'total eve charge','total night calls',
                      'total night charge','total intl charge']

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[
    ('encode', FunctionTransformer(encode)),
    ('log_transform', FunctionTransformer(log_transform)),
    ('drop_columns', FunctionTransformer(lambda df: df.drop(columns=columns_to_drop, errors='ignore'))),
    ('scaling', ColumnTransformer([
        ('scale', RobustScaler(), columns_to_scale)
    ], remainder='passthrough'))
])

X = df_copy.drop(columns=['churn'])  # Drop the target column from the feature set
y = df_copy['churn']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = X_train.copy()

# Fit the pipeline on training data
preprocessing_pipeline.fit(train_data)



# Load the trained best LightGBM model
best_lgb_model = joblib.load('best_lgb_model.pkl')  # Load the model (replace with correct path)

def predict_churn(new_data):
    """
    Predict whether a new customer will churn based on their feature values.

    Args:
    - new_data (pd.DataFrame): The new customer data

    Returns:
    - prediction (int): 1 if the customer is predicted to churn, 0 otherwise
    """
    # # Transform the new customer data
    transformed_data = preprocessing_pipeline.transform(new_data)

    column_order = ['account length', 'total day calls', 'total day charge',
                    'total eve calls', 'total eve charge','total night calls',
                    'total night charge', 'total intl charge', 'state',
                    'area code','international plan', 'voice mail plan',
        'binned_customer_service_calls', 'total_intl_calls_log']

    # Create DataFrame for transformed data and reindex according to the specified column order
    transformed_df = pd.DataFrame(transformed_data, columns=column_order)
    common_columns = new_data.columns.intersection(transformed_df.columns)
    transformed_df = transformed_df[common_columns]
    # print(transformed_df)

    preprocessed_data = transformed_df

    # Use the LightGBM model to predict churn
    prediction = best_lgb_model.predict(preprocessed_data)

    return prediction
