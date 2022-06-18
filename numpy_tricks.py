"""
Numpy is the core library for scientific computing in Python. It provides a high-performance multidimensional array
object, and tools for working with these arrays.

A numpy array is a grid of values, ALL OF THEM THE SAME TYPE (in contrast to list). Hence, the efficiency.

What Numpy provides:
- ndarray, a fast and space-efficient multidimensional array providing vectorized arithmetic operations and sophisticated broadcasting capabilities
- Standard mathematical functions for fast operations on entire arrays of data without having to write loops
- Tools for reading / writing array data to disk and working with memory-mapped files
- Linear algebra, random number generation, and Fourier transform capabilities
- Tools for integrating code written in C, C++, and Fortran

"""

import numpy as np

# =========================== Creating an Array ===========================
a = np.array([1, 2, 3])
a = np.asarray([1, 2, 3])

a[0] = 5  # Change an element of the array
print(a)  # Prints "[5, 2, 3]"
print(2*a)  # Prints "[50, 20, 30]"
print(2*[5, 2, 3]) # Prints "[5,2,3,5,2,3]"

# Nested sequences, like a list of equal-length lists, will be converted into a multidimen- sional array:
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
print(b[0, 0], b[0, 1], b[1, 0])  # Prints "1 2 5"

arr = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(arr)
print(arr2)  # array([[1, 2, 3, 4],
                    # [5, 6, 7, 8]])

print(np.array(b))
print(b.ndim)
print(b.shape)
print(b.dtype)
print(type(b))  # Prints "<class 'numpy.ndarray'>"

# =========================== Other functions to create arrays ===========================
a = np.zeros(10)  # Create an array of all zeros
print(a)  # Prints array([ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

a = np.zeros((2, 2))  # Create an array of all zeros
print(a)  # Prints "[[ 0.  0.]
          #          [ 0.  0.]]"

b = np.ones((1, 2))  # Create an array of all ones
print(b)  # Prints "[[ 1.  1.]]"

c = np.full((2, 2), 7)  # Create a constant array
print(c)  # Prints "[[ 7.  7.]
#          [ 7.  7.]]"

d = np.eye(2)  # Create a 2x2 identity matrix
d = np.identity(2)  # Create a 2x2 identity matrix
print(d)  # Prints "[[ 1.  0.]
#          [ 0.  1.]]"

e = np.random.random((2, 2))  # Create an array filled with random values
print(e)         # Might print "[[ 0.91940167  0.08143941]
#               [ 0.68744134  0.87236687]]"

k = np.random.randn(3,4) # might print array([[ 0.16462368,  0.40667506,  0.88176286,  0.73727787],
                                    #       [ 1.03054928,  1.10905476, -1.41934872, -0.49739245],
                                    #       [-0.09689096, -1.82724465,  1.19913053, -1.22382975]])

# It’s not safe to assume that np.empty will return an array of all zeros. In many cases,
# it will return uninitialized garbage values.
f = np.empty((2, 3, 2))
print(f)

print(np.arange(10))    # [0, 1, 2, ..., 9]

# =========================== Array indexing ===========================
# Slicing: Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional,
# you must specify a slice for each dimension of the array:

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3]

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print(a[0, 1])   # Prints "2"
b[0, 0] = 77     # b[0, 0] is the same piece of data as a[0, 1]
print(a[0, 1])   # Prints "77" use .copy to avoid

# You can also mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the
# original array. Note that this is quite different from the way that MATLAB handles array slicing:

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Two ways of accessing the data in the middle row of the array.
# Mixing integer indexing with slices yields an array of lower rank,
# while using only slices yields an array of the same rank as the
# original array:
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"

# We can make the same distinction when accessing columns of an array:
col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape)  # Prints "[[ 2]
                             #          [ 6]
                             #          [10]] (3, 1)"

# =========================== Integer array indexing: ===========================

# One-dimensional arrays indexing

arr = np.arange(10) # Prints [0,1,...,9]
print(arr[5])  # Print 5
print(arr[5:8])  # Print [5,6,7]
arr[5:8] = 12
print(arr) # array([ 0, 1, 2, 3, 4, 12, 12, 12, 8, 9])

# An important first dis- tinction from lists is that array slices are views on the original array.
#  This means that the data is not copied, and any modifications to the view will be reflected in the source array:

arr = np.arange(10) # Prints [0,1,...,9]
arr_slice = arr[5:8]
# arr_slice = arr[5:8].copy()  # if you want to create a new object
arr_slice[1] = 12345
print(arr)   # Print  array[0,1,2,3,4,12,12345,12,8,9]

# Two-dimensional arrays indexing
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[2]) # array([7, 8, 9])
print(arr2d[0][2])  # 3
print(arr2d[0, 2]) # 3

# multidimensional arrays indexing
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3d[0])   # array([[1, 2, 3],
                  #  [4, 5, 6]])

arr3d[0] = 42
print(arr3d)    # [[[42 42 42]
                #  [42 42 42]]
                # [[ 7  8  9]
                #  [10 11 12]]]

# arr3d[1, 0]gives you all of the values whose indices start with(1, 0),forming a 1-dimensional array:
print(arr3d[1, 0]) # array([7, 8, 9])

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr2d[:2])   #[[1 2 3]
                   # [4 5 6]]
print(arr2d[:, 2])  # [3 6 9]
print(arr2d[1, [2, 3]])  # [5,6]
print(arr2d[:2, 1:])  # [[2 3]
                      # [5 6]]


# Integer array indexing
a = np.array([[1, 2], [3, 4], [5, 6]])

# An example of integer array indexing.
# The returned array will have shape (3,) and
print(a[[0, 1, 2], [0, 1, 0]])  # Prints "[1 4 5]"

# The above example of integer array indexing is equivalent to this:
print(np.array([a[0, 0], a[1, 1], a[2, 0]]))  # Prints "[1 4 5]"

# When using integer array indexing, you can reuse the same
# element from the source array:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# Equivalent to the previous integer array indexing example
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

# Create a new array from which we will select elements
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

print(a)  # prints "array([[ 1,  2,  3],
          #                [ 4,  5,  6],
          #                [ 7,  8,  9],
          #                [10, 11, 12]])"

# Create an array of indices
b = np.array([0, 2, 0, 1])

# Select one element from each row of a using the indices in b
print(a[np.arange(4), b])  # Prints "[ 1  6  7 11]"

# Mutate one element from each row of a using the indices in b
a[np.arange(4), b] += 10

print(a)  # prints "array([[11,  2,  3],
          #                [ 4,  5, 16],
          #                [17,  8,  9],
          #                [10, 21, 12]])

# =========================== Integer array indexing: ===========================
a = np.array([[1, 2], [3, 4], [5, 6]])

bool_idx = (a > 2)   # Find the elements of a that are bigger than 2;
                     # this returns a numpy array of Booleans of the same
                     # shape as a, where each slot of bool_idx tells
                     # whether that element of a is > 2.

print(bool_idx)      # Prints "[[False False]
                     #          [ True  True]
                     #          [ True  True]]"

# We use boolean array indexing to construct a rank 1 array
# consisting of the elements of a corresponding to the True values
# of bool_idx
print(a[bool_idx])  # Prints "[3 4 5 6]"

# We can do all of the above in a single concise statement:
print(a[a > 2])     # Prints "[3 4 5 6]"

names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
print(data[names == 'Bob'])
print(data[names == 'Bob', 2:])
print(data[~(names == 'Bob')])
mask = (names == 'Bob') & (names == 'Will')
print(data[mask])
data[data < 0] = 0

# ====================== Fancy Indexing ==========================
# Fancy indexing is a term adopted by NumPy to describe indexing using integer arrays. Suppose we had a 8 × 4 array:

arr = np.full((8, 4), 0)

for i in range(8):
    arr[i] = i

print(arr[[4, 3, 0, 6]])
print(arr[[-3, -5, -7], 0])
print(arr[[1, 5, 7, 2], [0, 3, 1, 2]]) # selects pairwise points in the matrix
print(arr[(1, 0), (2, 0)])
print(arr[[1, 5, 7, 2]][[0, 3, 1, 2]])  # selected rows and columns


# ==================== Transposing Arrays and Swapping Axes ==========
arr = np.arange(15).reshape((3, 5))
print(arr.T)
arr = np.random.randn(6, 3)
np.dot(arr.T, arr)
arr = np.arange(32).reshape((2, 2, 4, 2))

# ======================= Universal Functions: Fast Element-wise Array Functions =======================================

"""A universal function, or ufunc, is a function that performs elementwise operations on data in ndarrays. You can 
think of them as fast vectorized wrappers for simple functions that take one or more scalar values and produce one or 
more scalar results."""

arr = np.arange(10)
np.sqrt(arr)
np.exp(arr)

x = np.random.randn(8)
y = np.random.randn(8)
print(np.maximum(x, y))


arr = [1.5, 2, 3.1, -4.9]
print(np.modf(arr))   # (array([ 0.5,  0. ,  0.1, -0.9]), array([ 1.,  2.,  3., -4.]))

# Unary ufuncs
print(np.abs(arr)) # Compute the absolute value element-wise for integer, floating point, or complex values.
print(np.square(arr))
print(np.sqrt(arr))
print(np.exp(arr))
print(np.log10([1, 2, 3]))  # log, log10, log2
print(np.sign([1, -2, 3, 0]))  # [1, -1, 1, 0]
print(np.ceil([1.4, -2.1, 3.1, 0.1]))  # [1, -1, 1, 0]
print(np.floor([1.4, -2.1, 3.1, 0.1]))  # [1, -3, 3, 0]
print(np.rint([1.4, -2.1, 3.1, 0.1]))  # [ 1. -2.  3.  0.]  # Round elements to the nearest integer, preserving the dtype
print(np.modf([1.4, -2.1, 3.1, 0.1])) # Return fractional and integral parts of array as separate array
print(np.isnan([1.4, np.nan, 3.1, 0.1])) # [False,True, False, False] # Return boolean array indicating whether each value is NaN (Not a Number)
print(np.isfinite([np.inf, np.nan, 3.1, 0.1])) # Return boolean array indicating whether each element is finite (non-inf, non-NaN) or infinite, respectively
print(np.cos([30, 60, 90, 180]))  # cos, cosh, sin, sinh, tan, tanh, arccos, arccosh, arcsin, arcsinh, arctan, arctanh
print(np.logical_not([np.nan, 0, 90, 180]))  # cos, cosh, sin, sinh, tan, tanh, arccos, arccosh, arcsin, arcsinh, arctan, arctanh

# Binary universal functions
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 0, 2])

print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1/arr2)
print(arr1**arr2)
print(np.maximum(arr1, arr2))
print(np.fmax(arr1, arr2)) #ignore np.nan
print(np.minimum(arr1, arr2))
print(np.fmin(arr1, arr2))
print(np.mod(arr1, arr2))
print([arr1 >= arr2])
print(np.logical_or(arr1>0, arr2>0))
print(np.logical_and(arr1>0, arr2>0))

# ======================= Datatypes ===========================
x = np.array([1, 2])   # Let numpy choose the datatype
print(x.dtype)         # Prints "int64"

x = np.array([1.0, 2.0])   # Let numpy choose the datatype
print(x.dtype)             # Prints "float64"

x = np.array([1, 2], dtype=np.int64)   # Force a particular datatype
print(x.dtype)                         # Prints "int64"

a = np.array([1, 2, 3], dtype=np.float64)
print(a.astype(np.int64))   # [1., 2. 3.]

# when use astype, it is copied into a new object. Hence, assignment is required.
numeric_strings = np.array(['1.25', '-9.6', '42'])
numeric_strings = numeric_strings.astype(np.float64)   # [1.25, -9.6, 42]

# =========================== Array math ===========================
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

# Elementwise sum; both produce the array
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# Inverse
print(1/x + 1/y)

# Elementwise difference; both produce the array
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# Elementwise product; both produce the array
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# Elementwise division; both produce the array
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# Elementwise square root; produces the array
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

v = np.array([9, 10])
w = np.array([11, 12])

# Inner product of vectors; both produce 219
print(v.dot(w))
print(np.dot(v, w))

# Matrix / vector product; both produce the rank 1 array [29 67]
print(x.dot(v))
print(np.dot(x, v))

# Matrix / matrix product; both produce the rank 2 array
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

x = np.array([[1, 2], [3, 4]])

print(np.sum(x))  # Compute sum of all elements; prints "10"
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"

x = np.array([[1, 2], [3, 4]])
print(x)    # Prints "[[1 2]
            #          [3 4]]"
print(x.T)  # Prints "[[1 3]
            #          [2 4]]"

# Note that taking the transpose of a rank 1 array does nothing:
v = np.array([1, 2, 3])
print(v)    # Prints "[1 2 3]"
print(v.T)  # Prints "[1 2 3]"

# =========================== Broadcasting ===========================
# Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing
# arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array
# multiple times to perform some operation on the larger array.

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = np.empty_like(x)   # Create an empty matrix with the same shape as x

# Add the vector v to each row of the matrix x with an explicit loop
for i in range(4):
    y[i, :] = x[i, :] + v

# Now y is the following
# [[ 2  2  4]
#  [ 5  5  7]
#  [ 8  8 10]
#  [11 11 13]]
print(y)

# alternative approach and faster
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))   # Stack 4 copies of v on top of each other
print(vv)                 # Prints "[[1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]
                          #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print(y)  # Prints "[[ 2  2  4
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"

# Numpy broadcasting allows us to perform this computation without actually creating multiple copies of v.
# Consider this version, using broadcasting:
# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print(y)  # Prints "[[ 2  2  4]
          #          [ 5  5  7]
          #          [ 8  8 10]
          #          [11 11 13]]"




