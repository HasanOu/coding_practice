from collections import Counter
from collections import defaultdict
import itertools

# ========================================== all tricks on dictionary ==================================================
# update dictionary
dict1 = {1: 1, 2: 9, 3: 4}
dict2 = {1: 1, 3: 9, 4: 4, 5: 4}
dict1.update(dict2)
print(dict1)

# print key and value in a dictionary
dict1 = {1: 1, 2: 9, 3: 4}
for item in dict1.items():
    print(item)

for key in dict1.keys():
    print(key)

for value in dict1.values():
    print(value)

# change the key and value in a dictionary
dict2 = {}
for (key, value) in dict1.items():
    dict2[value] = key
print(dict2)

# sort dictionary
dict1 = {1: 1, 2: 9, 3: 4}
sorted_tuples = sorted(dict1.items(), key=lambda item: item[1])
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict = {k: v for k, v in sorted_tuples}
print(sorted_dict)  # {1: 1, 3: 4, 2: 9}

# sort dictionary by value decreasing order
dict1 = {1: 1, 2: 9, 3: 4}
sorted_tuples = sorted(dict1.items(), key=lambda item: -1*item[1])
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict = {k: v for k, v in sorted_tuples}
print(sorted_dict)  # {1: 1, 3: 4, 2: 9}

# sort dictionary by key decreasing order
dict1 = {1: 1, 2: 9, 3: 4}
sorted_tuples = sorted(dict1.items(), key=lambda x: -1*x[0])
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict = {k: v for k, v in sorted_tuples}
print(sorted_dict)  # {1: 1, 3: 4, 2: 9}

# invert a list dictionary
dict_1 = {1: [1, 2, 4], 2: [9, 6, 1], 3: [4, 2, 1]}

dict_2 = defaultdict(list)
for (key, value) in dict_1.items():
    for item_val in value:
        dict_2[item_val].append(key)
print(dict_2)

# reverse list in a dictionary
dict_1 = {1: [1, 2, 4], 2: [9, 6, 1], 3: [4, 2, 1]}
dict_2 = defaultdict(list)
for (key, value) in dict_1.items():
    sorted_value = sorted(value)
    dict_2[key].extend(sorted_value)
print(dict_2)

# concatenate dicts
dic1 = {1:10, 2:20}
dic2 = {3:30, 4:40}
dic3 = {5:50,6:60}

my_dict = {}
for d in [dic1, dic2, dic3]:
    my_dict.update(d)
print(my_dict)

# get the maximum and min key in dict
print(max(my_dict.keys()))

# get the maximum and min value in dict
print(max(my_dict.values()))

# sum values in dictionary
my_dict = {'data1': 100, 'data2': -54, 'data3': 247}
print(sum(my_dict.values()))

# Write a Python program to remove a key from a dictionary.
my_dict = {'a':1,'b':2,'c':3,'d':4}
print(my_dict)
if 'a' in my_dict:
    del my_dict['a']
print(my_dict)

#  Write a Python program to map two lists into a dictionary.
keys = ['red', 'green', 'blue']
values = ['#FF0000', '#008000', '#0000FF']
my_dict = dict(zip(keys, values))
print(my_dict)

# remove duplicate from dict
my_dict = {'data1': 100, 'data2': -54, 'data3': 247, 'data4': 247}
result = {}
for key, value in my_dict.items():
    if value not in result.values():
        result[key] = value
print(result)

# check empty dictionary
if not bool(my_dict):
    print("not empty")

# Write a Python program t combine two dictionary adding values for common keys.
d1 = {'a': 100, 'b': 200, 'c':300}
d2 = {'a': 300, 'b': 200, 'd':400}

my_dic = Counter(d1) + Counter(d2)
print(my_dic)

for key1, value1 in d1.items():
    for key2, value2 in d2.items():
        if key1 == key2:
            my_dic[key1] = value1+value2
        if key1 not in my_dic.keys():
            my_dic[key1] = value1
        if key2 not in my_dic.keys():
            my_dic[key2] = value2
print(my_dic)

# Write a Python program to print all unique values in a dictionary.
my_dict = {"V":"S001", "VI": "S001", "VII":"S005", "VIII":"S007", "V4": "S007"}
print(set(my_dict.values()))

# Write a Python program to create and display all combinations of letters, selecting each letter from a different key
# in a dictionary.
d ={'1':['a','b'], '2':['c','d']}
for combo in itertools.product(*[d[k] for k in sorted(d.keys())]):
    print(''.join(combo))

# Write a Python program to find the highest 3 values of corresponding keys in a dictionary
my_dict = {0: [1, 4, 100, 150], 1: [2, 3, 20, 25]}
target_key = 0
for key in my_dict.keys():
    if key == target_key:
        top_3_values = sorted(my_dict[key], reverse=True)[0:3]
print(top_3_values)

# Write a Python program to combine values in python list of dictionaries
my_list = [{'item': 'item1', 'amount': 400}, {'item': 'item2', 'amount': 300}, {'item': 'item1', 'amount': 750}]
my_dict = defaultdict(int)
for d in my_list:
    key = d['item']
    value = d['amount']
    my_dict[key] += value
print(my_dict)

# Write a Python program to create a dictionary from a string.
my_string = '"w3resource" "hello" '
my_dict = Counter(my_string)
print(my_dict)

# Write a Python program to count the values associated with key in a dictionary
student = [{'id': 1, 'success': True, 'name': 'Lary'},
 {'id': 2, 'success': False, 'name': 'Rabi'},
 {'id': 3, 'success': True, 'name': 'Alex'}]
print(sum(d['id'] for d in student))
print(sum(d['success'] for d in student))

# Write a Python program to count number of items in a dictionary value that is a list.\
my_dict = {'Alex': ['subj1', 'subj2', 'subj3'], 'David': ['subj1', 'subj2']}
print(list(map(len, my_dict.values())))

# Write a Python program to replace dictionary values with their average
import numpy as np
my_dict = {'Alex': [1, 3, 4], 'David': [1, 2]}
print(list(map(np.mean, my_dict.values())))

# ========================================== all tricks on list ==================================================

# Write a Python program to sum all the items in a list.
my_list = [1,2,3,4]
print(sum(my_list))

# Write a Python program to get the largest number from a list
my_list = [1,2,3,4]
print(max(my_list))

# Write a Python program to count the number of strings where the string length is 2 or more and the first and
# last character are same from a given list of strings
my_list = ['abc', 'xyz', 'aba', '1221']
print([item for item in my_list if item[0] == item[-1] and len(item) >= 2])

# Write a Python program to get a list, sorted in increasing order by the last element in
# each tuple from a given list of non-empty tuples.
my_list = [(2, 5), (1, 2), (4, 4), (2, 3), (2, 1)]
print(sorted(my_list, key=lambda x: x[1]))

#  Write a Python program to remove duplicates from a list.
my_list = [1,2,3,4,1,2,3,4]
print(list(set(my_list)))

# Write a Python program to clone or copy a list.
my_list = [1,2,3,4]
my_list_copy = my_list.copy()

# Write a Python function that takes two lists and returns True if they have at least one common member
my_list_1 = [1, 2, 3, 4]
my_list_2 = [1, 2, 3, 4]

if len(set(my_list_1).intersection(my_list_2)) >= 1:
    print("True")

# Write a Python program to print a specified list after removing the 0th, 4th and 5th elements.
my_list = ['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow']
new_list = []
removed_list_index = [0, 4, 5]
for i, x in enumerate(my_list):
    if i not in removed_list_index:
        new_list.append(x)
print(new_list)

#  Write a Python program to print the numbers of a specified list after removing even numbers from it

#  Write a Python program to get the cumulative sum of the elements of a given list
my_list = [1, 2, 3, 4]

cum_list = []
for i in range(1, len(my_list)+1):
    cum_list.append(sum(my_list[:i]))
print(cum_list)

# Given a list of integers nums and an integer k, write a Python function to find the kth largest element in the list.
my_list = [1,4,3,2,5]
k=3
kth_larger_elelemt = sorted(my_list)[k-1]

# =============================================================  Python tricks on tuple ================================
