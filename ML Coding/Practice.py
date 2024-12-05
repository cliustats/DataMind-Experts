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
print(df)
