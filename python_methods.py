# ======================================= python Built-in Functions

# Returns the absolute value of a number
print(abs(-2))

# Returns True if all items in an iterable object are true
print(all([True, 1, True]))

# Returns True if any item in an iterable object is true
print(any([1, True, 0]))

# Returns a readable version of an object. Replaces none-ascii characters with escape character
x = ascii("My name is Ståle")
print(x)

# Returns the binary version of a number
print(bin(2))

# Returns the boolean value of the specified object
print(bool(1))

# Returns an array of bytes
print(bytearray(4))

# Returns a bytes object
print(bytes(4))


# Returns True if the specified object is callable, otherwise False
def x():
    a = 5


print(callable(x))

# Returns a character from the specified Unicode code.
print(chr(97))

# Returns a complex number
print(complex(3, 5))

# Returns a dictionary (Array)
x = dict(name="John", age=36, country="Norway")
print(x)

# Returns the quotient and the remainder when argument1 is divided by argument2
print(divmod(5, 2))

print(5 // 2, 5 % 2)

# Takes a collection (e.g. a tuple) and returns it as an enumerate object
for (index, item) in enumerate(('apple', 'banana', 'cherry')):
    print(index, item)
for (index, item) in enumerate(['apple', 'banana', 'cherry']):
    print(index, item)


# Eval Evaluates and executes an expression
def max_(a, b):
    return max(a, b)


x = 'max_(1,5)'
print(eval(x))

# Executes the specified code (or object)
print(exec(x))


# Use a filter function to exclude items in an iterable object
# Returns the specified iterator with the specified function applied to each item
def add7(x):
    return x + 7


def isOdd(x):
    return x % 2 == 0


a = list(range(11))
print(list(map(add7, filter(isOdd, a))))

# Returns a floating point number
print(float(3))

# Freeze the list, and make it unchangeable:
mylist = ['apple', 'banana', 'cherry']
print(frozenset(mylist))

# Converts a number into a hexadecimal value
print(hex(5))

# Allowing user input
print(f"print your user name:")
# my_username = input()
# print('Hello, ' + my_username)

# Returns an integer number
int(5)

# Returns an iterator object
x = iter(["apple", "banana", "cherry"])
print(next(x))
print(next(x))

# Returns the length of an object
print(len([1, 2, 3]))
print(len((1, 2, 3)))
print(len("hello"))

# Returns a list
print(list({1, 2, 3}))


# Returns the specified iterator with the specified function applied to each item
def myfunc(n):
    return len(n)


x = list(map(myfunc, ['apple', 'banana', 'cherry']))
print(x)

# max and min Returns the largest item in an iterable
print(max([1, 2, 3]))
print(max((1, 2, 3)))
print(max({1, 2, 3}))

# Returns the next item in an iterable
x = iter(("Hello", "Bye"))
next(x)


# Returns the value of x to the power of y
print(pow(2, 3))

# Prints to the standard output device
print("hello")

# Returns a sequence of numbers, starting from 0 and increments by 1 (by default)
print(range(1,10,2))

for i in range(1, 10, 2):
    print(i)

# Returns a reversed iterator
for i in reversed((1, 2, 3)):
    print(i)

a= [1, 2, 3]
print(a[::-1])


# Rounds numbers
print(round(5.76543, 2))

x = set(['apple', 'banana', 'cherry'])
print(x)


a = ("a", "b", "c", "d", "e", "f", "g", "h")
print(a[slice(2, 7, 2)])

# Returns a sorted list
sorted([1, 4, 2])

# Returns a string object
print(str([1, 23])[0])

# Sums the items of an iterator
print(sum((1,2,3)))

# Returns a tuple
print(tuple([1, 2, 3]))

# Returns the type of an object
print(type(tuple([1,2,3])))
print(type(5))

# Returns an iterator, from two or more iterators
a = ("John", "Charles", "Mike")
b = ("Jenny", "Christy", "Monica")
c = (1, 2, 3)

x = zip(a, b, c)
for i in x:
    print(i)

