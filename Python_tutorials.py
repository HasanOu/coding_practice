# Source https://www.youtube.com/watch?v=k9TUPpGqYTo&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=2&ab_channel=CoreySchafer

# ======================== How to work with text data

"""
This is a comment
"""

message = "Hello\'s World"
print('Hello World')

print(len(message))
print(message[0], message[1],message[-1])
print(message[1:4])
print(message[:4])
print(message[4:])

print(message.lower())
print(message.upper())
print(message.capitalize())
print(message.count('Hello'))
print(message.find('ld'))
new_message = message.replace("l", "b")
print(message)
print(new_message)

greeting = "Hello"
name = "Hasan"
message = f"{greeting} {name}"
print(message)
print(dir(message))
print(help(str.strip))

# ==================== How to work with Numeric data
num = 3.14
num = int("314")
print(type(num))
print(3/2)
print(3//1.9) # the integer part of it (Ceil)
print(3**2)
print(3%2)

print(round(1.46,1))

print(1!=2)
print(1==2)

# =============== List, Tuples, and Sets
courses = ['course1', 'course2', 'course3']

courses.append('course4') # inplace
courses.extend(['course5', 'course6']) #inplace
courses.insert(3, 'course7') # inplace
courses = courses + ['course8']
print(courses)

courses = ['course1', 'course1', 'course3', 'course4']
courses.remove('course1') # removes the first item with this value
print(courses)
courses.pop() # remove only the last item
print(courses)

courses.reverse()
courses.sort(reverse=True)
print(courses)

courses.index('course1')
print('course1' in courses)

new_courses = sorted(courses)
print(new_courses)

min([1,2,3])
max([1,2,4])
sum([1,2,3])

for item in courses:
    print(item)

courses = ['c1', 'c2', 'c3', 'c4']
for index, course in enumerate(courses[1]):
    print(index, course)

course_str = ",".join(courses)
print(course_str)
new_list = course_str.split(",")
print(new_list)

# ====================================== tuple ========================================
tuple_1 = ('History', 'Math')
# tuple_1[1] = 'Hello' # does not work

# =================================== set ============================================
# set does membership more efficiently
# set does give duplicate values

cs_courses = {'History', 'Math', 'Physics', 'ComSci', 'History'}
art_courses = {'History'}
print(cs_courses)
print('History' in cs_courses)
cs_courses.intersection(art_courses)
cs_courses.union(art_courses)
cs_courses.difference(art_courses)
empty_set = set()

# =================================== Dictionaries (hash map) ========================
student = {'name': 'John', 'age': 25, 'courses': ['Math', 'ComSci']}
print(student)
# keys can be any immutable
print(student['name'])
print(student['courses'])

print(student.get('name'))
print(student.get('phone', 'Not found'))
student.update({'name': 'Jane', 'age': 26, 'phone': '555-5555'})
print(student)
del student['age']
print(student)
age = student.pop('phone')
print(student)

len(student)
print(student.keys())
print(student.values())
print(student.items())

# ===================================== Conditionals and Booleans - If, Else, and Elif Statements
language = 'Python'
user = 'Hasan'

# ==, <, <=, >, >=, !=, is
# Object identity: is (checks if two objects are the same)

# and or not

if not (language == 'Python' and user == 'Hasan'):
    print("This was true")
elif language == 'Java':
    print("This was true")
else:
    print('NA')

a = [1,2,3]
b = [1, 2, 3]

print(a==b)   # True
print(a is b)  # False

a = b
print(a is b)  # True

print(id(a))
print(id(b))

# False values
# False
# None
# Zero of any numeric type
# An empty sequence. For example, '', (), [].
# Any empty mapping. For example {}

condition = False
condition = 'Test'

if condition:
    print('Evaluated to True')
else:
    print('Evaluated to False')

# ====================================== Loops and iterations ========================
nums = [1,2,3,4]

for num in nums:
    print(num)

# break will break out of the loop
for num in nums:
    if num ==3:
        print("we found it")
        break
    print(num)

# continue will go ahead to the next iterations
for num in nums:
    if num == 3:
        print("we found it")
        continue
    print(num)

for num in nums:
    for letter in 'abc':
        print(num, letter)

for index in range(1, 11):
    print(index)

index = 0
while index < 11:
    index += 1
    print(index)

x = 0
while True:
    if x == 5:
        break
    print(x)
    x+=1

# =================================== Functions =============
# Some instruction packed together to do a specific task
# Keeping your code dry (i.e., don't repeat ypurself over and over again)

def hello_func(greeting = 'Hello'):
    print(greeting + "hello World")
    return greeting + "hello world"

print(hello_func) # gives the function object
print(hello_func().upper()) # gives the function value return
print(hello_func("Welcome").upper()) # gives the function value return

# *args: gives a tuple for all positional arguments
# *kwargs: gives a dictionary with all keyword values

def student_info(*args, **kwargs):
    print(args)
    print(kwargs)

student_info('Math', 'Art', name='John', age=22)

def student_info(*args, **kwargs):
    print(args)
    print(kwargs)

courses = ['Math, Art']
info = {'name': 'John', 'age': 22}
student_info(*courses, **info)

# ===================================== Import Modules and Exploring The Standard Library
# we create it in another file and import my_module
# import my_module as mm
# from my_module import my_func, my_var
# from my_module import my_func as m_func
# from my_module import * (don't use it)

import sys

print(sys.path)
# sys.path.append("Add your path")
# alternatively you can add this path in your computer depending on mac or windows (system variable)

import os
import datetime
import calendar
import random
import math

print(os.getcwd())
print(os.__file__) # it shows the entire standard lib in your system

# =======================  Setting up a Python Development Environment in Sublime Text


# =======================  Setting up a Python Development Environment in Atom

# =======================  Setting up a Python Development Environment in Eclipse

# =======================  Python Tutorial: pip - An in-depth look at the package management system
# In terminal
# pip help
# pip help install
# pip search Numpy
# pip install numpy
# pip list
# pip uninstall numpy
# pip list --outdated (to check the latest version)
# pip list -o (to check the latest version)
# pip freeze
# pip freeze > requirments.txt
# cat requirments.txt
# pip install -r requirments.txt
# pip list --outdated
# This updates all packages to the latest version

# ========================= virtual ENV ================================
# the goal is that each project will have its own packages and versions
# In terminal
# mkdir Environments
# pip install virtualenv
# pip list

# =========================== Anaconda


# ======================== How to manage Multiple Projects, Virtual Env, and Env variables


# ================= Jupyter Notebook


# ================= Variable scope ==========================================
# LEGB Local, Enclosing, Global, Built-in

# working with local make it easier to work with
# global makes it difficult because we need to worry about overwrite variable outside of that function
# we need to define many many variables

# built-in is just built-in in python
# you can overwrite the built-in function (local function overwrites the built-in function)

import builtins
print(dir(builtins))

x = 'global x' # it is global because it is in the main body
z = 'global z'

def test():
    y = 'local y'
    global x
    x = 'local x'
    z = 'local z'
    print(y)
    print(x)

test()
print(x)
print(z)

# enclosing:
# it is very similar to local and global scope. Just follow that logic.

print ("*********")
x = 'global x'
def outer():
    x = 'outer x'
    def inner():
        # nonlocal x  # it works similar to global within this function
        x = 'inner x'
        print(x)

    inner()
    print(x)

outer()
print(x)


# ===================== slicing list and string ======================
my_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
print(my_list[1])
print(my_list[-1])
print(my_list[2:8:3])
print(my_list[1:-1:3])
print(my_list[1:])
print(my_list[1:])
print(my_list[-1:2:-1])
print(my_list[::-1])

sample_url = 'helloworld'
print(sample_url[::-1])
print(sample_url[1:-1:2])
print(sample_url[7:])

#  Python Tutorial: Comprehensions - How they work and why you should be using them
nums = [1,2,3,4,5]
my_list = [n for n in nums]
my_list = [n**2 for n in nums]
my_list = map(lambda n: n**n, nums)
print(my_list)

my_list = [n for n in nums if n%2 == 0]
my_list = list(map(lambda n: n**2, filter(lambda n: n%2 == 0, nums)))
print(my_list)

my_list = [(letter, num) for letter in 'abcd' for num in range(4)]
print(my_list)

names = ['Tannaz', 'Hasan', 'Arash']
heros = ['Fatemieh', 'Manzour', 'Tarkhan']
print(list(zip(names, heros)))

my_dict = {name: hero for name, hero in zip(names, heros) if name != 'Arash'}
print(my_dict)

nums = [1,1,2,1,2,3,4,5,6]
my_set = {n**2 for n in nums}
print(my_set)

# Generator Expression
nums = [1,2,3,4,5,6,7]
def gen_func(nums):
    for n in nums:
        yield n*n

my_gen = gen_func(nums)

for i in my_gen:
    print(i)

# Generator expression
my_gen = (n*n for n in nums)

for i in my_gen:
    print(i)

# ======================= How to sort List, Tupels, and Objects ==============
li = [1,4,2,1]
s_li = sorted(li, reverse =True) # creates a new object
li.sort(reverse=True) # sort the original
print(s_li)

# sort method works specifically on list
# sorted function works on any iterable

new_tuple = sorted((1,4,3,2))
dict = {'name': 'Corey', 'job': 'programming', 'age': None}
s_di = sorted(dict) # sorted keys by default

li = [-6, -5, -4, -3, -2, -1]
s_di = sorted(li, key = abs)
print(s_di)

dict = {'name': 'Corey', 'job': 'programming'}
ss_di = sorted(dict.items(), key=lambda item: item[1])
print("***")
print(ss_di)

def my_sort_func(x):
    return -x

my_list = [1,2,3,4]
ss_di = sorted(my_list, key=my_sort_func, reverse= True)
ss_di = sorted(my_list, key=lambda x: -x, reverse= True)

print("***")
print(ss_di)

# ================================ String Formatting ====================
sentence = f'My name is {0} and I am {1}'
my_date = datetime.datetime(2016, 9, 24, 12, 30, 45)
sentence = '{0:%B %d %Y} fell on a {0:%A} and was the {0:%j} of the year'.format(my_date)
print(sentence)

# ======================================= OS module =========
# navigate file
#import os
#print(dir(os))
#print(os.getcwd())
#print(os.chdir(os.getcwd()))
#print(os.listdir())
#print(os.makedirs('new_folder/Sub_dirc'))
#print(os.removedirs('remove_dir'))
#os.rename('test.txt')
# os.stat('demo.txt')
# print(os.walk())
# print(os.path.join(path, file_name))
# print(os.path.join(os.environ.get('HOME'), file_name))
# os.path.basename()
# os.path.exists('path_name')
# os.path.isdir('path')
# os.path.isfile('path')

# ============================== How to work with datetime ==========================
import datetime

# naive datetime: they dont have enough info for zone and etc

d = datetime.date(2016, 8, 24)
print(d)

today = datetime.date.today()
print(today)
print(today.year)
print(today.month)
print(today.weekday())
print(today.isoweekday())

timedelta = datetime.timedelta(days=7)
print(today + timedelta)

tday = datetime.date.today()
bday = datetime.date(2016, 9, 24)
til_day = bday - tday
print(til_day.days)
print(til_day.total_seconds())

#
print("******")
dt = datetime.datetime(2016, 7, 26)
print(dt.day)
print(dt.year)
print(dt.isoformat())

print(datetime.datetime.today())
print(datetime.datetime.now())
print(datetime.datetime.utcnow())

import pytz

dt = datetime.datetime(2016, 7, 27, 3, 45, tzinfo = pytz.UTC)
dt_now = datetime.datetime.now(tz = pytz.UTC)
print(dt_now)

dt_mnt = dt_now.astimezone(pytz.timezone('US/Pacific'))
print(dt_mnt)

# =================================== How to work with file objects ================
# File objects
# f = open('test.txt', 'r+')
# print(f.name)
# print(f.mode)
# print(f.close())
#
# with open('test.txt', 'r') as f:
#     for line in f:
#         print(line, end='')
#
#     size_to_read = 100
#     while len(f_contents) > 0:
#         print(f_contents, end='')
#         f_content = f.read(size_to_read)
#
# print(f.read())

# ===========


# =========== random module
import random

val = random.random()
print(val)

val = random.uniform(1,5)
print(val)

val = random.randint(4,10)
print(val)

greetings = ['Hello', 'Hi', 'Ho']
value = random.choice(greetings)
print(value + ', Hasan')

colors = ['red', 'black', 'green']
results = random.choices(colors, weights = [18, 18, 2], k=10)
print(results)

deck = list(range(1,53))

random.shuffle(deck)
print(deck)

hand = random.sample(deck, k=5)
print(hand)

# ========================= how to read and parse CSV file ============

# ==========================

# ========================== regular expression
import re

# raw string:

pattern = re.compile(r'abc')

test_to_search = "hello abc"
matches = pattern.finditer(test_to_search)

for match in matches:
    print(match)

# ================================ Python Error Hnadling
# try:
#     pass
# except FileNotFoundError as e:
#     print(e)
# except Exception as e:
#     print(e)
# else:
#     # print(f.read())
#     # f.close()
# finally:
#     print("asdas") # this will always executes regardless if try or else got executed
#     # forexample when u want to close the database

# ================================= How to be pythonic ============
# Duck Typing
# EAFP (Easier to ask forgivness than permission)

# Duck Typing is a way of programming in which an object passed into a function or method supports all
# method signatures and attributes expected of that object at run time. The object's type itself is not important.
# Rather, the object should support all methods/attributes called on it.
# "If it walks like a duck and it quacks like a duck, then it must be a duck"

# =============================== First class function =================
# A programming language is said to have First-class functions when functions in that
# language are treated like any other variable.
# For example, in such a language, a function can be passed as an argument
# to other functions, can be returned by another function and can be assigned as a value to a variable

def square(x):
    return x*x

f = square
print(square)
print(f(5))

# higher order function
# map function takes a function and array as its argument

def my_map(func, array):
    result = []
    for i in array:
        result.append(func(i))
    return result

squares = my_map(square, [1,2,3,4,5])
print(squares)

# returning a function from another function (higher order function)
def logger(msg):
    def log_message():
        print('Log:', msg)

    return log_message

log_hi = logger("Hi")
log_hi()

# =================================== Closure ===========================
# Python closure is a nested function that allows us to access variables of the outer function even after
# the outer function is closed.

# A closure is inner function that remembers and has access to the variable and scope of which it was created even after
# the outer function is finished executing
def outer_func(msg):
    message = msg

    def inner_func():
        print(message)

    # return inner_func()
    return inner_func

hi_func = outer_func("hasan")
my_func = outer_func("tannaz")  # this is now inner_func

hi_func()
my_func()

# =================================================== Decorators ==================
# # In Python, a decorator is a design pattern that allows you to modify the behavior of a function or a class without
# # changing its source code. A decorator is itself a function that takes another function as input, adds some
# # functionality to it, and returns a new function that can be used in place of the original function.
# #
# # To use a decorator, you define a function that implements the desired behavior and decorate the target function with the decorator function using the "@" symbol. Here's an example:
#
# def my_decorator(func):
#     def wrapper():
#         return func()
#     return wrapper
#
# @my_decorator
# def display():
#     print('display func ran')
#
# #decorated_display = my_decorator(displace)
# #decorated_display()
#
# display()

# =========================================
print("***********")
def validate_input(func):
    def wrapper(x, y):
        if x <= 0 or y <= 0:
            return "x and y must be positive."
        else:
            return func(x, y)
    return wrapper

@validate_input
def add_numbers(x, y):
    return x + y

result = add_numbers(3, 4)
print(result)

result = add_numbers(-2, 5)  # Raises a ValueError
print(result)



original = [1,2, 2, 3]
new = original.copy()
new[0] = 10
print(original)
print(new)


import pandas as pd

# Create two DataFrames
df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})
df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': [5, 6, 7, 8]})

# Merge the two DataFrames based on the 'key' column
merged = pd.merge(df1, df2, on='key')

print(merged)

import pandas as pd

# Create a DataFrame with some sample data
df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                   'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                   'C': [1, 2, 3, 4, 5, 6, 7, 8],
                   'D': [10, 20, 30, 40, 50, 60, 70, 80]})

print(df)
# Crea
# te a pivot table that groups by column 'A' and aggregates by the sum of column 'C'
pivot_table = pd.pivot_table(df, values='C', index='A', aggfunc='sum')

print(pivot_table)

import pandas as pd

# create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [5, 6, 7, 8]})

import pandas as pd

# create a DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4],
                   'B': [5, 6, 7, 8]})

# define a function to calculate the mean
def mean(x):
    return x.mean()

# group the DataFrame by column A and calculate the mean of column B for each group
df_mean = df.groupby('A')['B'].transform(mean)

print(df_mean)



































































