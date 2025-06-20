"""
This is a list of conceptual questions in Python
"""

# 1. What is the Difference Between a Shallow Copy and Deep Copy?

# 2. What is Python Global Interpreter Lock?

# 3. What Advantage Does the Numpy Array Have over a Nested List?

# 4. What are Pickling and Unpickling?

# 5. How is Memory managed in Python?

# 6. What is Python interpreter?

# 7. What is lazy evaluation?

# 8. Are Arguments in Python Passed by Value or by Reference?

# 9 What is the Purpose of the Pass Statement?

# 10 What Is the Difference Between Del and Remove() on Lists?

# 11.  Differentiate Between append() and extend().

# 12 What Is the Difference Between a List and a Tuple?

# 13 What Is Docstring in Python?

# 14 How Do You Use the Split() Function in Python?

# 15 Is Python Object-oriented or Functional Programming?

# 16 What is functional programming?
# Immutability
# Pure function
# Higher order function

# 17 What is a pure function?

# 18 Can you give an example of impure function?

# 19 What is a higher-order function?

# 20 What Are *args and *kwargs?

# 21 “in Python, Functions Are First-class Objects.” What Do You Infer from This?

# 22 What Is the Output Of: Print(__name__)? Justify Your Answer.

# 23 How do you access a CSV file in Goole sheet using Python?

# 24 What Is the Difference Between range() and xrange() Functions in Python?

# 25 Which Python Library Is Built on Top of Matplotlib and Pandas to Ease Data Plotting?

# 26 What are the important features of Python?

# 27 What is PEP 8?

# 28 Explain Python namespace.
#Built-in
#Global
#Enclosing
#Local

# 29 What is closure?

# 30 What are decorators in Python?

def my_decorator(func):
    def wrapper():
        print("Before the function is called")
        func()
        print("after the func is called")
    return wrapper

@my_decorator
def say_hello():
    print("hello")

say_hello()

# 31 What is the differentiate between .pyc and .py.

# 32 Explain global variables and local variables in Python.

# 33 What is the use of self in Python?

# 34  What are the literals in Python?

# 35 What are types of literals in Python?

# 36 What is the difference between expression and statement?

# 37 What is the different between iterator and iteration?
# iterator ---> generator or iter([1,2,3])

# 38 What are Python modules? Name a few Python built-in modules that are often used.

# 39  What is _init_?

# 40 What is the Lambda function?

# 41  How does continue, break, and pass work?

# 42 What are Python iterators?

# 43  Differentiate between range and xrange.

# 44 What are unpickling and pickling?

# 45 What are generators in Python?

# 46 What are the functions help() and dir() used for in Python?

# 47 What is a dictionary in Python?

# 48 Explain the split(), sub(), and subn() methods of the Python "re" module.
# regular expression

# 49  What are Python libraries?

# 50 Explain monkey patching in Python.

# 51 What is inheritance in Python?

# 52 What are the different types of inheritance in Python?

# 53 Explain polymorphism in Python.
# oBJECTS CAN GET DIFFERENT FORM DEPENDING ON THE CONTEXT (INPUT)

# 54 What is encapsulation in Python?
# make use of _, __ to make each method or attribute private or protected
# _ private __ protected

# 55  What is GIL?

# 56 How to write a Unicode string in Python?

# 67 What is the difference between staticmethod and classmethod in Python?

# 68 How do you handle exceptions in Python? What are some common built-in exceptions?

# 69 What is a generator in Python? How do you create and use them?

# 70 How do you handle concurrency and parallelism in Python? What are some common libraries for this purpose?

# 71 What are Python's built-in data structures? How do you choose which one to use for a given task?

# 72 How do you serialize and deserialize data in Python? What are some common serialization formats?

# 73 What are some common design patterns in Python? How do you use them?

# 74 What are some common anti-patterns in Python? How do you avoid them?

# 75 How do you check if a string is empty?

# 76 How do ypu check if a file exists?

# 77 How do you read a file line by line?

# 78 How do you write a list to a file?

# 79 How do you read data froma  database?

# 80 What are the key modules in Python for database

