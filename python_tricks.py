
# sort dictionary by value increasing order

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
sorted_tuples = sorted(dict1.items(), key=lambda item: -1*item[0])
print(sorted_tuples)  # [(1, 1), (3, 4), (2, 9)]
sorted_dict = {k: v for k, v in sorted_tuples}

print(sorted_dict)  # {1: 1, 3: 4, 2: 9}


# play with matrix
matrix = [[1,2,3,4],
     [5,6,7,8]]

print([[row[i] for row in matrix] for i in range(len(matrix[0]))])


a = [1,2,34]

print(a.index(4))