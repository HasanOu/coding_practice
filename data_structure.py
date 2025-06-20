# ========================================================
#   Python Cheat-Sheet: STRING Tips, Tricks & Idioms
# ========================================================

import string
from string import Template

# 1. String Creation & Multiline
print("\n--- 1. Creation & Multiline ---")
s1 = 'Hello'
s2 = "World"
s3 = '''This is a 
multiline string.'''
print(s1, s2)
print(s3)

# 2. String Concatenation & Repetition
print("\n--- 2. Concatenation & Repetition ---")
print("Concat         :", s1 + " " + s2)
print("Repeat         :", "Ha" * 3)

# 3. String Formatting
print("\n--- 3. Formatting Strings ---")
name = "Alice"
age = 30
print(f"f-string       : Hello, {name}. You are {age}.")
print("format()       : Hello, {}. You are {}.".format(name, age))

# 4. Indexing, Slicing, Stepping
print("\n--- 4. Indexing & Slicing ---")
word = "Python"
print("First letter   :", word[0])
print("Last letter    :", word[-1])
print("Slice 1:4      :", word[1:4])
print("Reversed       :", word[::-1])

# List slicing examples for clarity
my_list = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]
print(my_list[1])
print(my_list[-1])
print(my_list[2:8:3])
print(my_list[1:-1:3])
print(my_list[1:])
print(my_list[::-1])

sample_url = 'helloworld'
print(sample_url[::-1])
print(sample_url[1:-1:2])
print(sample_url[7:])

# 5. Case Methods
print("\n--- 5. Case Methods ---")
text = "hello world"
print("Upper          :", text.upper())
print("Capitalize     :", text.capitalize())
print("Title Case     :", text.title())
print("Swapcase       :", text.swapcase())

# 6. Search & Replace
print("\n--- 6. Search & Replace ---")
s = "banana"
print("Count 'a'      :", s.count('a'))
print("Find 'na'      :", s.find('na'))
print("Replace        :", s.replace('na', 'HA'))

# 7. Starts/Ends With
print("\n--- 7. Starts/Ends With ---")
url = "https://openai.com"
print("Starts with 'https'?", url.startswith("https"))
print("Ends with '.com'?   ", url.endswith(".com"))

# 8. Strip & Clean
print("\n--- 8. Strip & Clean ---")
messy = "   hello world!   "
print("Strip whitespace   :", messy.strip())
print("Strip left         :", messy.lstrip())
print("Strip right        :", messy.rstrip())

# 9. Split & Join
print("\n--- 9. Split & Join ---")
csv = "apple,banana,cherry"
print("Split by ','       :", csv.split(','))
words = ['join', 'these', 'words']
print("Join with space    :", ' '.join(words))

# 10. Testing String Content
print("\n--- 10. Content Testing ---")
val = "abc123"
print("Is alpha?          :", val.isalpha())
print("Is digit?          :", val.isdigit())
print("Is alnum?          :", val.isalnum())
print("Is lower?          :", val.islower())
print("Is upper?          :", val.isupper())
print("Is title?          :", "Hello World".istitle())

# 11. String Constants & Character Sets
print("\n--- 11. String Module Constants ---")
print("ASCII Letters      :", string.ascii_letters)
print("Digits             :", string.digits)
print("Punctuation        :", string.punctuation)

# 12. Useful String Utilities
print("\n--- 12. Useful String Utilities ---")
print("Zero-filled        :", "42".zfill(5))
print("Centered           :", "hi".center(10, '-'))
print("Left-justified     :", "hi".ljust(10, '.'))
print("Right-justified    :", "hi".rjust(10, '.'))

# 13. Encoding & Decoding
print("\n--- 13. Encoding & Decoding ---")
utf8 = "Café".encode('utf-8')
print("Encoded (utf-8)    :", utf8)
decoded = utf8.decode('utf-8')
print("Decoded            :", decoded)

# 14. Replace Multiple Spaces / Strip Extra
print("\n--- 14. Strip Extra Spaces ---")
dirty = "this   is   messy  "
cleaned = " ".join(dirty.split())
print("Cleaned            :", cleaned)

# 15. Reversing Word Order
print("\n--- 15. Reverse Word Order ---")
sentence = "Python is awesome"
reversed_words = " ".join(reversed(sentence.split()))
print("Reversed order     :", reversed_words)

# 16. Counting Words & Characters
print("\n--- 16. Counting Words ---")
text = "the quick brown fox jumps"
words = text.split()
print("Word count         :", len(words))
print("Character count    :", len(text.replace(" ", "")))

# 17. Remove All Punctuation
print("\n--- 17. Remove Punctuation ---")
sentence = "Hello, world! How's it going?"
no_punct = ''.join(c for c in sentence if c not in string.punctuation)
print("No punctuation     :", no_punct)

# 18. Template Strings (for substitution)
print("\n--- 18. Template Strings ---")
t = Template("Hello $name, you are $age.")
print(t.substitute(name="John", age=40))

# ========================================================
#   Python Cheat-Sheet: LIST Tips, Tricks & Idioms
# ========================================================

from collections import Counter, deque
from copy import deepcopy
import itertools

# 1️⃣ Creation, Comprehension, and Filtering
print("\n--- 1. Creation & Comprehension ---")
squares = [x * x for x in range(1, 8)]
evens = [x for x in range(10) if x % 2 == 0]
print("Squares:", squares)
print("Evens  :", evens)

# Count strings with same first and last character
my_list = ['abc', 'xyz', 'aba', '1221']
filtered = [item for item in my_list if item[0] == item[-1] and len(item) >= 2]
print("Same first/last character:", filtered)

# Remove specific indices from list
my_list = ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
indices_to_remove = [0, 4, 5]
cleaned_list = [x for i, x in enumerate(my_list) if i not in indices_to_remove]
print("Removed index 0, 4, 5:", cleaned_list)

# Clone a list
my_list = [1, 2, 3, 4]
my_list_copy = my_list.copy()
print("Cloned list:", my_list_copy)

# Check if two lists share a common element
my_list_1 = [1, 2, 3, 4]
my_list_2 = [5, 6, 3, 7]
common = bool(set(my_list_1) & set(my_list_2))
print("Have common elements?", common)

# Cumulative sum
my_list = [1, 2, 3, 4]
cum_sum = [sum(my_list[:i+1]) for i in range(len(my_list))]
print("Cumulative sum:", cum_sum)

# kth largest element
my_list = [1, 4, 3, 2, 5]
k = 3
kth_largest = sorted(my_list, reverse=True)[k-1]
print(f"{k}th largest element:", kth_largest)

# 2️⃣ Slicing & Stepping
print("\n--- 2. Slicing & Stepping ---")
letters = ['a', 'b', 'c', 'd', 'e', 'f']
print("Slice 1:4          :", letters[1:4])
print("Every 2nd element  :", letters[::2])
print("Reversed (slicing) :", letters[::-1])

# 3️⃣ Adding & Removing Elements
print("\n--- 3. Adding & Removing ---")
lst = ['x', 'y']
lst.append('z')
lst.extend(['u', 'v'])
lst.insert(0, 'w')
print("After additions     :", lst)
lst.pop()
lst.remove('w')
print("After removals      :", lst)

# 4️⃣ Enumerate, Zip, Unpack
print("\n--- 4. Enumerate, Zip, Unpack ---")
for idx, val in enumerate(['p', 'q', 'r'], start=1):
    print(f"Index {idx} ⇒ {val}")

nums = [1, 2, 3]
words = ['one', 'two', 'three']
print("Zipped pairs        :", list(zip(nums, words)))

first, *middle, last = [10, 20, 30, 40, 50]
print("Unpacked parts      :", first, middle, last)

# 5️⃣ Sorting
print("\n--- 5. Sorting ---")
people = [('Alice', 25), ('Bob', 19), ('Cara', 30)]
print("Sorted by age ASC   :", sorted(people, key=lambda x: x[1]))
print("Sorted by age DESC  :", sorted(people, key=lambda x: x[1], reverse=True))

strings = ['aa', 'b', 'cccc']
print("Sorted by length ↓  :", sorted(strings, key=len, reverse=True))

# 6️⃣ Flattening & Transposing
print("\n--- 6. Flatten & Transpose ---")
matrix = [[1, 2, 3], [4, 5, 6]]
flat = [num for row in matrix for num in row]
transposed = list(map(list, zip(*matrix)))
print("Flattened matrix    :", flat)
print("Transposed matrix   :", transposed)

# 7️⃣ Shallow vs Deep Copy
print("\n--- 7. Shallow vs Deep Copy ---")
orig = [[1, 2], [3, 4]]
shallow = orig[:]
deep = deepcopy(orig)
orig[0][0] = 99
print("Shallow copy        :", shallow)
print("Deep copy           :", deep)

# 8️⃣ List Multiplication Pitfall
print("\n--- 8. List Multiplication Pitfall ---")
bad_2d = [[0] * 3] * 4
good_2d = [[0] * 3 for _ in range(4)]
bad_2d[0][0] = 1
good_2d[0][0] = 1
print("Bad 2D list         :", bad_2d)
print("Good 2D list        :", good_2d)

# 9️⃣ Stack vs Queue
print("\n--- 9. Stack & Queue ---")
stack = []
stack.append('eat')
stack.append('code')
print("Stack pop           :", stack.pop())

queue = deque(['eat', 'code', 'sleep'])
queue.append('repeat')
print("Queue popleft       :", queue.popleft())

# 🔟 Built-ins: any, all, min, max, sum
print("\n--- 10. Built-in Functions ---")
nums = [2, 4, 6]
print("Any odd?            :", any(n % 2 for n in nums))
print("All even?           :", all(n % 2 == 0 for n in nums))
print("Min/Max/Sum         :", min(nums), max(nums), sum(nums))

# 1️⃣1️⃣ Frequency Counts & Deduplication
print("\n--- 11. Frequency & Deduplication ---")
data = ['a', 'b', 'a', 'c', 'b', 'a']
freq = Counter(data)
print("Frequency counts    :", freq)

seen, dedup = set(), []
for x in data:
    if x not in seen:
        dedup.append(x)
        seen.add(x)
print("Deduped list        :", dedup)

# 1️⃣2️⃣ Negative Indexing & Joining
print("\n--- 12. Negative Indexing & Join ---")
items = ['alpha', 'beta', 'gamma']
print("Last item           :", items[-1])
print("Joined string       :", ', '.join(items))

# 1️⃣3️⃣ Chunking a List
print("\n--- 13. Chunking List ---")
def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]

print("Chunks of 2         :", list(chunks(list(range(7)), 2)))

# 1️⃣4️⃣ Cartesian Product
print("\n--- 14. Cartesian Product ---")
prod = list(itertools.product([1, 2], ['a', 'b']))
print("Cartesian product   :", prod)

pair =  list(zip([1, 2], ['a', 'b']))

# ========================================================
#   Python Cheat-Sheet: Tuples Tips, Tricks & Idioms
# ========================================================

# 1️⃣ Tuple Basics
print("\n--- 1. Tuple Basics ---")
t1 = (1, 2, 3)
t2 = 1, 2, 3                # Parentheses optional
singleton = (5,)           # Important: comma makes it a tuple
not_a_tuple = (5)          # Just an int

print("Tuple t1         :", t1)
print("Singleton        :", singleton)
print("Type check       :", type(singleton), type(not_a_tuple))

# 2️⃣ Indexing, Slicing, Length
print("\n--- 2. Indexing, Slicing ---")
t = ('a', 'b', 'c', 'd', 'e')
print("First element     :", t[0])
print("Last element      :", t[-1])
print("Slice [1:4]       :", t[1:4])
print("Length            :", len(t))

# 3️⃣ Tuple Packing & Unpacking
print("\n--- 3. Packing & Unpacking ---")
a, b, c = (1, 2, 3)
print("Unpacked          :", a, b, c)

# Extended unpacking (Python 3+)
first, *middle, last = (10, 20, 30, 40, 50)
print("First/Middle/Last :", first, middle, last)

# Swap values using tuple unpacking
x, y = 5, 10
x, y = y, x
print("Swapped           :", x, y)

# 4️⃣ Tuple Operations
print("\n--- 4. Tuple Operations ---")
t1 = (1, 2)
t2 = (3, 4)
t3 = t1 + t2               # Concatenation
print("Concatenated      :", t3)

t4 = t1 * 3                # Repetition
print("Repeated          :", t4)

# 5️⃣ Membership, Count, Index
print("\n--- 5. Membership, Count, Index ---")
t = (1, 2, 3, 2, 4, 2)
print("Is 3 in tuple?     :", 3 in t)
print("Count of 2         :", t.count(2))
print("Index of 4         :", t.index(2))

# 6️⃣ Tuples as Dictionary Keys
print("\n--- 6. Tuples as Dict Keys ---")
coords = {(0, 0): "origin", (1, 2): "point A"}
print("Dictionary access  :", coords[(1, 2)])

# 7️⃣ Tuple Sorting & Conversion
print("\n--- 7. Tuple Sorting & Conversion ---")
unsorted = [(2, 'b'), (1, 'a'), (3, 'c')]
sorted_by_first = sorted(unsorted)
sorted_by_second = sorted(unsorted, key=lambda x: x[1])
print("Sorted by index 0  :", sorted_by_first)
print("Sorted by index 1  :", sorted_by_second)

# Convert between tuple and list
tup = (1, 2, 3)
lst = list(tup)
tup2 = tuple(lst)
print("List → Tuple → List:", tup, lst, tup2)

# 8️⃣ Nested Tuples
print("\n--- 8. Nested Tuples ---")
nested = ((1, 2), (3, 4), (5, 6))
for a, b in nested:
    print(f"Pair: {a} + {b} = {a + b}")

# 9️⃣ Named Tuples (advanced readability)
print("\n--- 9. Named Tuples ---")
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p1 = Point(2, 3)
print("NamedTuple         :", p1)
print("Access by name     :", p1.x, p1.y)

# 1️⃣0️⃣ Use Case: Returning Multiple Values
print("\n--- 10. Returning Multiple Values ---")
def stats(numbers):
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

lo, hi, avg = stats([4, 8, 15, 16, 23, 42])
print("Min/Max/Avg        :", lo, hi, avg)

# ========================================================
#   Python Cheat-Sheet: Hashmap/Dict Tips, Tricks & Idioms
# ========================================================

from collections import defaultdict, Counter
import itertools
import numpy as np

# 1️⃣ Updating and Merging Dictionaries
print("\n--- 1. Updating and Merging Dictionaries ---")
dict1 = {1: 1, 2: 9, 3: 4}
dict2 = {1: 10, 4: 5}
dict1.update(dict2)
print("Updated dict1:", dict1)

d1 = {1: 10}
d2 = {2: 20}
d3 = {3: 30}
merged = {}
for d in [d1, d2, d3]:
    merged.update(d)
print("Merged:", merged)

# 2️⃣ Iterating Over Keys, Values, and Items
print("\n--- 2. Iterating Over Keys, Values, and Items ---")
my_dict = {1: 'a', 2: 'b', 3: 'c'}
for k in my_dict:
    print("Key:", k)
for v in my_dict.values():
    print("Value:", v)
for k, v in my_dict.items():
    print(f"Item: {k} => {v}")

# 3️⃣ Reversing or Inverting Keys and Values
print("\n--- 3. Reversing or Inverting Keys and Values ---")
original = {1: 'a', 2: 'b', 3: 'c'}
inverted = {v: k for k, v in original.items()}
print("Inverted:", inverted)

# 4️⃣ Sorting Dictionaries
print("\n--- 4. Sorting Dictionaries ---")
d = {1: 40, 2: 10, 3: 30}
sorted_by_val = dict(sorted(d.items(), key=lambda x: x[1]))
print("Sorted by val (asc):", sorted_by_val)

sorted_by_val_desc = dict(sorted(d.items(), key=lambda x: -x[1]))
print("Sorted by val (desc):", sorted_by_val_desc)

sorted_by_key_desc = dict(sorted(d.items(), key=lambda x: -x[0]))
print("Sorted by key (desc):", sorted_by_key_desc)

# 5️⃣ Dictionary with List Values: Inversion and Aggregation
print("\n--- 5. Dictionary with List Values: Inversion and Aggregation ---")
d = {1: [10, 20], 2: [20, 30]}
inv = defaultdict(list)
for k, vals in d.items():
    for v in vals:
        inv[v].append(k)
print("Inverted list dict:", dict(inv))

sorted_dict = {k: sorted(v) for k, v in d.items()}
print("Sorted list values:", sorted_dict)

# 6️⃣ Aggregation and Stats
print("\n--- 6. Aggregation and Stats ---")
d = {1: 40, 2: 10, 3: 30}
print("Max key:", max(d))
print("Max value:", max(d.values()))
print("Sum of values:", sum(d.values()))

scores = {'Alice': [80, 90], 'Bob': [70, 75, 65]}
avg_scores = {k: np.mean(v) for k, v in scores.items()}
print("Average scores:", avg_scores)

# 7️⃣ Remove Elements or Duplicates
print("\n--- 7. Remove Elements or Duplicates ---")
d = {'a': 1, 'b': 2}
d.pop('a', None)
print("After removal:", d)

d = {'a': 1, 'b': 2, 'c': 1}
unique = {}
for k, v in d.items():
    if v not in unique.values():
        unique[k] = v
print("After duplicate removal:", unique)

# 8️⃣ Map or Combine Lists into Dictionaries
print("\n--- 8. Map or Combine Lists into Dictionaries ---")
keys = ['red', 'green', 'blue']
vals = ['#FF0000', '#00FF00', '#0000FF']
color_map = dict(zip(keys, vals))
print("Color map:", color_map)

data = [{'item': 'apple', 'count': 3}, {'item': 'banana', 'count': 2}, {'item': 'apple', 'count': 5}]
result = Counter()
for d in data:
    result[d['item']] += d['count']
print("Item counts:", result)

# 9️⃣ Unique Values and Combinations
print("\n--- 9. Unique Values and Combinations ---")
d = {"A": "x", "B": "y", "C": "x"}
print("Unique values:", set(d.values()))

d = {'1': ['a', 'b'], '2': ['x', 'y']}
for combo in itertools.product(*d.values()):
    print("Combo:", ''.join(combo))

# 🔟 Dictionary from a String or Count by Condition
print("\n--- 10. Dictionary from a String or Count by Condition ---")
s = "hello world"
print("Character frequency:", Counter(s))

students = [{'pass': True}, {'pass': False}, {'pass': True}]
print("Pass count:", sum(d['pass'] for d in students))

# 1️⃣1️⃣ Count List Items in Dict Values
print("\n--- 11. Count List Items in Dict Values ---")
d = {'Alice': ['math', 'eng'], 'Bob': ['bio', 'chem', 'phys']}
counts = {k: len(v) for k, v in d.items()}
print("Subject counts:", counts)


# ========================================================
#   Python Cheat-Sheet: Set Tips, Tricks & Idioms
# ========================================================

from collections import Counter

# 🔹 1. Basic Creation & Properties
print("\n--- 1. Creation & Properties ---")
s1 = {1, 2, 3, 2}                   # Duplicates auto-removed
s2 = set([3, 4, 5])                 # From list
print("s1 =", s1, "| len:", len(s1))
print("s2 =", s2)

# 🔹 2. Adding & Removing Elements
print("\n--- 2. Adding & Removing Elements ---")
s1.add(4)
s1.update([5, 6])                   # Add multiple elements
print("After add/update:", s1)

s1.discard(6)                       # Safe remove (no error if absent)
removed = s1.pop()                  # Remove & return arbitrary element
print("After discard & pop:", s1, "(popped:", removed, ")")

# 🔹 3. Membership & Subset Tests
print("\n--- 3. Membership & Subset Tests ---")
print("3 in s1?", 3 in s1)
print("Is s1 subset of s2?", s1.issubset(s2))
print("Is s1 superset of {3,4}?", s1.issuperset({3, 4}))

# 🔹 4. Set Operations: Union, Intersection, Difference, Symmetric Difference
print("\n--- 4. Set Operations ---")
A, B = {1, 2, 3}, {3, 4, 5}
print("Union (A ∪ B):", A | B)
print("Intersection (A ∩ B):", A & B)
print("Difference (A − B):", A - B)
print("Symmetric Difference (A ⊕ B):", A ^ B)

# 🔹 5. Removing Duplicates from a List
print("\n--- 5. Deduplicate List ---")
nums = [1, 1, 2, 3, 2, 4]
unique_nums = list(set(nums))
print("Deduped list:", unique_nums)

# 🔹 6. Set Comprehension
print("\n--- 6. Set Comprehension ---")
squares = {x * x for x in range(6)}
print("Squares set:", squares)

# 🔹 7. Frozen Sets (Hashable, Immutable)
print("\n--- 7. Frozen Sets ---")
immutable = frozenset({1, 2, 3})
mapping = {immutable: "I am frozen!"}
print("Frozen as dict key:", mapping[immutable])

# 🔹 8. Filtering with Sets (O(1) Lookups)
print("\n--- 8. Filtering with Sets ---")
words = ["apple", "banana", "cherry", "date"]
vowels = set("aeiou")
starts_with_vowel = [w for w in words if w[0] in vowels]
print("Starts with vowel:", starts_with_vowel)


arr = [4, 1, 2, 2, 3, 3, 3, 4, 4, 4]
# 🔹 10. Set-Based Duplicate Detection
print("\n--- 10. Duplicate Detection ---")
seen, dupes = set(), set()
for x in arr:
    if x in seen:
        dupes.add(x)
    else:
        seen.add(x)
print("Duplicates found:", dupes)

# ========================================================
#   Python Cheat-Sheet: namedtuple Tips, Tricks & Idioms
# ========================================================

from collections import namedtuple

# 1️⃣ Basic Usage
print("\n--- 1. Basic Usage ---")
Point = namedtuple('Point', ['x', 'y'])
p1 = Point(3, 4)
print("NamedTuple        :", p1)
print("Access by field   :", p1.x, p1.y)
print("Access by index   :", p1[0], p1[1])

# 2️⃣ Auto-Renaming Invalid Field Names
print("\n--- 2. Auto-Renaming Invalid Field Names ---")
Record = namedtuple('Record', ['id', 'class', 'def'], rename=True)
r = Record(1, 'math', 'value')
print("Renamed fields    :", r._fields)
print("Values            :", r)

# 3️⃣ Introspection: _fields and _asdict()
print("\n--- 3. Introspection with _fields and _asdict() ---")
print("Field names       :", p1._fields)
print("As dictionary     :", p1._asdict())

# 4️⃣ Updating Values with _replace()
print("\n--- 4. _replace() for Updates ---")
p2 = p1._replace(x=10)
print("Original          :", p1)
print("Updated           :", p2)

# 5️⃣ Iteration & Unpacking
print("\n--- 5. Iteration & Unpacking ---")
for val in p1:
    print("Iterated value    :", val)

x, y = p1
print("Unpacked          :", x, y)

# 6️⃣ Setting Defaults using Prototype + _replace
print("\n--- 6. Setting Defaults ---")
Person = namedtuple('Person', ['name', 'age', 'city'])
default_person = Person('Unknown', 0, 'Nowhere')
p = default_person._replace(name='Alice', age=30)
print("Default person    :", default_person)
print("Modified person   :", p)

# 7️⃣ Creating from Dictionary
print("\n--- 7. Creating from Dictionary ---")
data = {'x': 5, 'y': 9}
p = Point(**data)
print("From dictionary   :", p)

# 8️⃣ Why Use namedtuple?
print("\n--- 8. Advantages of namedtuple ---")
print("- Lightweight and memory-efficient alternative to classes")
print("- Immutable and hashable (usable as dict keys)")
print("- Ideal for read-only structured data (e.g., records, rows, points)")

# ========================================================
#   Python Cheat-Sheet: deque Tips, Tricks & Idioms
# ========================================================

from collections import deque

# 1️⃣ Creation
print("\n--- 1. Creation ---")
dq1 = deque()                       # empty
dq2 = deque([1, 2, 3])              # from iterable
print("Initial deque       :", dq2)

# 2️⃣ Append / Append Left
print("\n--- 2. Append Operations ---")
dq = deque()
dq.append(10)                      # adds to right
dq.appendleft(5)                   # adds to left
print("After appends       :", dq)

# 3️⃣ Pop / Pop Left
print("\n--- 3. Pop Operations ---")
dq.pop()                           # remove from right
dq.append(20)
dq.appendleft(1)
left = dq.popleft()               # remove from left
print("After pops          :", dq)
print("Left popped value   :", left)

# 4️⃣ Rotation
print("\n--- 4. Rotation ---")
dq = deque([1, 2, 3, 4, 5])
dq.rotate(2)                      # right shift
print("Right rotated by 2  :", dq)
dq.rotate(-3)                     # left shift
print("Left rotated by 3   :", dq)

# 5️⃣ Max Length (bounded queue)
print("\n--- 5. Max Length / Bounded deque ---")
dq = deque(maxlen=3)
for i in range(5):
    dq.append(i)
    print(f"After appending {i}: {dq}")

# 6️⃣ Clear, Extend, Extend Left
print("\n--- 6. Other Operations ---")
dq = deque([1, 2])
dq.extend([3, 4])                 # adds to right
dq.extendleft([0, -1])           # adds to left in reverse order
print("After extend(s)     :", dq)
dq.clear()
print("After clear         :", dq)

# 7️⃣ Indexing and Reversing
print("\n--- 7. Indexing & Reversing ---")
dq = deque(['a', 'b', 'c'])
print("First element       :", dq[0])
dq.reverse()
print("After reverse       :", dq)

# 8️⃣ Convert between list and deque
print("\n--- 8. Conversion ---")
lst = [1, 2, 3]
dq = deque(lst)
back_to_list = list(dq)
print("List → deque        :", dq)
print("deque → List        :", back_to_list)

# 9️⃣ Use Case: Sliding Window
print("\n--- 9. Sliding Window Pattern ---")
nums = [1, 3, 5, 7, 9, 2, 6]
window = deque(maxlen=3)
for n in nums:
    window.append(n)
    print(f"Window: {list(window)} | Sum: {sum(window)}")

# 🔠 Use Case: Palindrome Check
print("\n--- 10. Palindrome Check ---")
def is_palindrome(s):
    d = deque(s)
    while len(d) > 1:
        if d.popleft() != d.pop():
            return False
    return True

print("Is 'racecar' a palindrome?", is_palindrome("racecar"))
print("Is 'banana' a palindrome? ", is_palindrome("banana"))

print("\n============= END OF deque CHEAT SHEET =============")

# ========================================================
#   Python Cheat-Sheet: heapq (Heap) Tips & Tricks
# ========================================================

from heapq import (
    heappush, heappop, heapify, heapreplace,
    heappushpop, nsmallest, nlargest, merge
)

# 1️⃣ Creating & Initialising a Min-Heap
print("\n--- 1. Build / Heapify ---")
data = [8, 3, 5, 1, 4]
heapify(data)              # in-place O(n)
print("Min-heap:", data)

# 2️⃣ Push & Pop
print("\n--- 2. Push / Pop ---")
heappush(data, 0)
print("After push 0 :", data)
smallest = heappop(data)
print("Popped min   :", smallest, "| heap:", data)

# 3️⃣ heappushpop vs heapreplace
print("\n--- 3. pushpop / replace ---")
print("Push-then-pop :", heappushpop(data, 2))
print("Heap now      :", data)
print("Replace top   :", heapreplace(data, 7))
print("Heap after    :", data)

# 4️⃣ nsmallest / nlargest
print("\n--- 4. k Smallest / Largest ---")
nums = [9, 1, 8, 2, 7, 3]
print("3 smallest:", nsmallest(3, nums))
print("2 largest :", nlargest(2, nums))

# 5️⃣ Merging Sorted Iterables
print("\n--- 5. Merge Sorted Streams ---")
a = [1, 4, 9]
b = [2, 3, 8]
print("Merged:", list(merge(a, b)))

# 6️⃣ Storing (priority, item) Pairs
print("\n--- 6. Tuples for Priority Queues ---")
tasks = []
heappush(tasks, (2, "write docs"))
heappush(tasks, (1, "fix bug"))
heappush(tasks, (3, "release"))
print("Popped by priority:", heappop(tasks))

# 7️⃣ Max-Heap Patterns
print("\n--- 7. Max-Heap (two tricks) ---")
nums = [4, 1, 7]
max_heap = [-n for n in nums]
heapify(max_heap)
print("Max:", -heappop(max_heap))

# or store (–priority, item)
tasks = []
heappush(tasks, (-5, "low prio"))
heappush(tasks, (-1, "high prio"))
print("Highest priority:", heappop(tasks)[1])

# 8️⃣ Stream Median (two-heap sketch)
print("\n--- 8. Median of Stream (concept) ---")
low, high = [], []  # max-heap (negated) & min-heap

def add(num):
    heappush(low, -num)
    heappush(high, -heappop(low))
    if len(high) > len(low):
        heappush(low, -heappop(high))

def median():
    return (-low[0] if len(low) > len(high)
            else (-low[0] + high[0]) / 2)

for n in [5, 2, 8, 3, 9]:
    add(n)
    print("Added", n, "median→", median())

print("\n============= END OF HEAP CHEAT SHEET =============")



