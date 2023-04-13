import collections
import numpy as np

"""Given an array of integers nums and an integer target, return indices of the
two numbers such that they add up to target.

You may assume that each input would have exactly one solution,
and you may not use the same element twice.

You can return the answer in any order.

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""
import collections

import numpy as np

def twoSum(nums, target):
    # create a dictionary to store each number and its index
    nums_dict = {}

    # loop through each number in the array
    for i in range(len(nums)):

        # calculate the complement (the number we need to reach the target)
        complement = target - nums[i]

        # if the complement is already in the dictionary, we found a solution
        if complement in nums_dict:
            return [nums_dict[complement], i]

        # if the complement is not in the dictionary, add the current number and its index to the dictionary
        nums_dict[nums[i]] = i

    # if no solution was found, return an empty array
    return []

arr = [1, 5, 4, 4, 7, 9]
output = twoSum(arr, 10)
print(output)
print("***********")


"""Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

    Open brackets must be closed by the same type of brackets.
    Open brackets must be closed in the correct order.
    Every close bracket has a corresponding open bracket of the same type.

s = "()[]{}"
Output = True 
"""

def IsValid(my_string):
    stack = []
    my_list = list(my_string)

    my_dict = {"(": ")", "[": "]", "{": "}"}

    for item in my_list:
        stack.append(item)
        if stack[-1] in [")", "]", "}"]:
            if stack[-1] == my_dict[stack[-2]]:
                stack = stack[:-2]
            else:
                return False

    if len(stack) == 0:
        return True
    else:
        return False

my_string = "()[]{}"
print(IsValid(my_string))
print("*******")

# Merge Two Sorted Lists
"""
You are given the heads of two sorted linked lists list1 and list2.

Merge the two lists in a one sorted list. The list should be made by splicing together the nodes of the first two lists.

Return the head of the merged linked list.

Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
"""

def MergeLists(list1, list2):
    if len(list1) == 0:
        return list2

    if len(list2) == 0:
        return list1

    MergeList = []
    while min(len(list1), len(list2)) != 0:
        if list1[0] < list2[0]:
            MergeList.append(list1[0])
            list1 = list1[1:]
        else:
            MergeList.append(list2[0])
            list2 = list2[1:]

    return MergeList + list1 + list2

list1 = [1,2,4]
list2 = [2,3,5,8]
print(MergeLists(list1, list2))
print("*****")

""" Best Time to buy stocks
You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
"""

def buy_sell_profit(my_list):
    start = 0
    end = len(my_list)-1

    max_profit = max(0, my_list[end] - my_list[start])

    if my_list[start] > my_list[start+1]:
        start += 1
        max_profit = my_list[end] - my_list[start]

    if my_list[end] < my_list[end-1]:
        end -= 1
        max_profit = my_list[end] - my_list[start]

    start += 1
    end -= 1
    max_profit = max(max_profit, my_list[end] - my_list[start])

    return max_profit

# Another implementation
def maxProfit(prices):
    # Initialize the minimum price to the first price in the array
    min_price = prices[0]

    # Initialize the maximum profit to 0
    max_profit = 0

    # Loop through the prices array starting from the second price
    for i in range(1, len(prices)):
        # Update the minimum price seen so far
        min_price = min(min_price, prices[i])
        # Calculate the profit that could be made by selling on this day
        current_profit = prices[i] - min_price
        # Update the maximum profit seen so far
        max_profit = max(max_profit, current_profit)

    # Return the maximum profit
    return max_profit

prices = [7,1,15,3,16,4]
print(buy_sell_profit(prices))
print("*******")

"""A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

Given a string s, return true if it is a palindrome, or false otherwise."""


def Validpalindrome(my_string):
    my_string = my_string.replace(" ", "").lower()

    if my_string[::-1] == my_string:
        return True
    else:
        return False


"""
Given the root of a binary tree, invert the tree, and return its root.
"""

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invertTree(root):
    # Base case: if the root is None, return None
    if not root:
        return None

    # Recursively invert the left and right subtrees
    left = invertTree(root.left)
    right = invertTree(root.right)

    # Swap the left and right subtrees
    root.left = right
    root.right = left

    # Return the inverted tree
    return root








"""You are given an integer array height of length n.
There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.

Notice that you may not slant the container.

Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water
(blue section) the container can contain is 49.
"""


def container_with_most_water(arr):
    max_area = 0
    left = 0
    right = len(arr) - 1
    while left < right:
        width = right - left
        max_area = max(max_area, width * min(arr[right], arr[left]))

        if arr[left] <= arr[right]:
            left += 1
        else:
            right -= 1

    return max_area


# arr = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# output = container_with_most_water(arr)
# print(output)

"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, 
and nums[i] + nums[j] + nums[k] == target.

Notice that the solution set must not contain duplicate triplets.

Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
"""


def threesum(arr, target):
    hashmap = {}
    threesum_list = []
    for i in range(len(arr)):
        hashmap[arr[i]] = i
        complement = target - arr[i]
        twosum_list = twoSum(arr, complement)

        for twosum in twosum_list:
            if i not in twosum:
                threesum_list.append(sorted([i] + twosum))

    threesum_list_ = []
    for item in threesum_list:
        if item not in threesum_list_:
            threesum_list_.append(item)

    threesum_list_value = [[arr[num] for num in group] for group in threesum_list_]
    return threesum_list_value


# arr = [3,3,1,2,-1,4]
# output = threesum(arr, 6)
# print(output)

""" Given an integer array nums of length n and an integer target, find three 
integers in nums such that the sum is closest to target.

Return the sum of the three integers.

You may assume that each input would have exactly one solution.

Input: nums = [-1,2,1,-4], target = 1
Output: 2
Explanation: The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
"""

def threesum_closest(arr, target):
    min_diff = float('inf')

    arr.sort()
    for i in range(len(arr)):
        lo, hi = i + 1, len(arr) - 1

        while (lo < hi):
            sum = arr[i] + arr[lo] + arr[hi]
            diff = sum - target
            if abs(diff) < abs(min_diff):
                min_diff = diff

            if sum < target:
                lo += 1
            else:
                hi -= 1

    return target + min_diff


# arr = [-1,4,1,-4]
# output = threesum_closest(arr, 1)
# print(output)

"""Given an array nums of n integers, return an array of all the unique quadruplets [nums[a], nums[b], nums[c], nums[d]] such that:

    0 <= a, b, c, d < n
    a, b, c, and d are distinct.
    nums[a] + nums[b] + nums[c] + nums[d] == target

You may return the answer in any order.

Input: nums = [1,0,-1,1,-2,2], target = 0
Output: [[-2,-1,1,2],[-2,0,0,2],[-1,0,0,1]]
"""


def foursum(arr, target):
    arr.sort()
    list_values = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            lo, hi = j + 1, len(arr) - 1

            while lo < hi:
                sum = arr[i] + arr[j] + arr[lo] + arr[hi]
                if sum == target:
                    list_values.append([arr[i], arr[j], arr[lo], arr[hi]])

                if sum <= target:
                    lo += 1
                else:
                    hi -= 1

    return list_values


output = foursum([1, 0, -1, 1, -2, 2], 0)
print(output)

"""
Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element 
appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array 
in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if 
there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. 
It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) 
extra memory.

nums = [0,0,1,1,1,2,2,3,3,4]
5, nums = [0,1,2,3,4,_,_,_,_,_]
"""


def remove_duplicates_from_sorted_array(arr):
    new_array = []
    for item in arr:
        if item not in new_array:
            new_array.append(item)

    remainder_size = len(arr) - len(new_array)

    remainder_array = ["_" for i in range(remainder_size)]

    new_array = new_array + remainder_array

    return new_array


arr = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
output = remove_duplicates_from_sorted_array(arr)
print(output)

"""
A permutation of an array of integers is an arrangement of its members into a sequence or linear order.

    For example, for arr = [1,2,3], the following are considered permutations of arr: [1,2,3], [1,3,2], [3,1,2], [2,3,1].

The next permutation of an array of integers is the next lexicographically greater permutation of its integer. More formally, if all the permutations of the array are sorted in one container according to their lexicographical order, then the next permutation of that array is the permutation that follows it in the sorted container. If such arrangement is not possible, the array must be rearranged as the lowest possible order (i.e., sorted in ascending order).

    For example, the next permutation of arr = [1,2,3] is [1,3,2].
    Similarly, the next permutation of arr = [2,3,1] is [3,1,2].
    While the next permutation of arr = [3,2,1] is [1,2,3] because [3,2,1] does not have a lexicographical larger rearrangement.

Given an array of integers nums, find the next permutation of nums.

The replacement must be in place and use only constant extra memory.
"""


def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1

    # Find the first index j from right to left such that nums[j] > nums[i]
    j = len(nums) - 1
    while j >= 0 and nums[j] <= nums[i]:
        j -= 1

    # Swap nums[i] and nums[j]
    nums[i], nums[j] = nums[j], nums[i]

    # Reverse the suffix starting at index i+1
    nums[i + 1:] = reversed(nums[i + 1:])


arr = [2, 3, 1, 4]
output = next_permutation(arr)
print(output)

"""
here is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) 
such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, 
or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
"""

def search_in_rotated_sorted_array(nums, target):
    # Initilize two pointers
    begin = 0
    end = len(nums) - 1
    while begin <= end:
        mid = (begin + end) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] > nums[end]:  # Left side of mid is sorted
            if nums[begin] <= target and target < nums[mid]:  # Target in the left side
                end = mid - 1
            else:  # in right side
                begin = mid + 1
        else:  # Right side is sorted
            if nums[mid] < target and target <= nums[end]:  # Target in the right side
                begin = mid + 1
            else:  # in left side
                end = mid - 1
    return -1


arr = [7, 8, 0, 1, 2, 3, 4]
output = search_in_rotated_sorted_array(arr, 7)
print(output)

""" Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the 
following rules:
    Each row must contain the digits 1-9 without repetition.
    Each column must contain the digits 1-9 without repetition.
    Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.

Note:
    A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    Only the filled cells need to be validated according to the mentioned rules.

Input: board = 
[["5","3",".",".","7",".",".",".","."]
,["6",".",".","1","9","5",".",".","."]
,[".","9","8",".",".",".",".","6","."]
,["8",".",".",".","6",".",".",".","3"]
,["4",".",".","8",".","3",".",".","1"]
,["7",".",".",".","2",".",".",".","6"]
,[".","6",".",".",".",".","2","8","."]
,[".",".",".","4","1","9",".",".","5"]
,[".",".",".",".","8",".",".","7","9"]]
Output: true
"""


def valid_sodoku(arr):
    for i in range(9):
        for j in range(9):
            if arr[i][j] == ".":
                arr[i][j] = f"{(i, j)}"

    for i in range(9):
        if len(set(arr[i])) != len(arr[i]):
            return False

    for j in range(9):
        col_arr = [arr[i][j] for i in range(9)]
        if len(set(col_arr)) != len(col_arr):
            return False

    for i in range(6):
        for j in range(6):
            List = []
            for x in np.arange(i, i + 2):
                for y in np.arange(j, j + 2):
                    item = arr[x][y]
                    if item in List:
                        return False
                    List.append(arr[x][y])
    return True


def sodoku1(arr):
    # Use hash set to record the status

    N = 9
    rows = [set() for _ in range(N)]
    cols = [set() for _ in range(N)]
    boxes = [set() for _ in range(N)]

    for r in range(N):
        for c in range(N):
            val = arr[r][c]
            # Check if the position is filled with number
            if val == ".":
                continue

            # Check the row
            if val in rows[r]:
                return False
            rows[r].add(val)

            # Check the column
            if val in cols[c]:
                return False
            cols[c].add(val)

            # Check the box
            idx = (r // 3) * 3 + c // 3
            if val in boxes[idx]:
                return False
            boxes[idx].add(val)

    return True


arr = [["5", "3", ".", ".", "7", ".", ".", ".", "."]
    , ["6", ".", ".", "1", "9", "5", ".", ".", "."]
    , [".", "9", "8", ".", ".", ".", ".", "2", "."]
    , ["8", ".", ".", ".", "6", ".", ".", ".", "3"]
    , ["4", ".", ".", "8", ".", "3", ".", ".", "1"]
    , ["7", ".", ".", ".", "2", ".", ".", ".", "6"]
    , [".", "6", ".", ".", ".", ".", "2", "8", "."]
    , [".", ".", ".", "4", "1", "9", ".", ".", "5"]
    , [".", ".", ".", ".", "8", ".", ".", "7", "9"]]
output = sodoku1(arr)
print(output)

"""
Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
"""

def Maximum_Subarray(arr):
    max_sum = 0
    temp_sum = 0
    List = []
    for val in arr:
        temp_sum += val
        List = List + [val]

        if temp_sum > max_sum:
            max_sum = temp_sum
            best_list = List

        if temp_sum < 0:
            temp_sum = 0
            List = []

    return max_sum, best_list


# arr = [4, 1, 2, -1]
# output = Maximum_Subarray(arr)
# print(output)

"""
Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals, 
and return an array of the non-overlapping intervals that cover all the intervals in the input.

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
"""


def merge_interval(arr):
    arr.sort(key=lambda x: x[0])

    merged = []
    for item in arr:
        if not merged or merged[-1][1] < item[0]:
            merged.append(item)
        else:
            merged[-1][1] = max(merged[-1][1], item[1])

    return merged


arr = [[0, 4], [5, 7], [2, 3]]
output = merge_interval(arr)
print(output)

"""
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

Input: nums = [1,1,1], k = 2
Output: 2

Input: nums = [1,2,3], k = 3
Output: 2
"""


def sum_k(arr, k):
    count = 0
    for start_value in range(len(arr)):
        sum = 0
        for end_value in np.arange(start_value, len(arr)):
            sum += arr[end_value]

            if sum == k:
                count += 1

    return count


arr = [1, 2, 3, 1, 6]
k = 6
output = sum_k(arr, k)
print(output)

"""
A website domain "discuss.leetcode.com" consists of various subdomains. At the top level, we have "com", at the next 
level, we have "leetcode.com" and at the lowest level, "discuss.leetcode.com". When we visit a domain like 
"discuss.leetcode.com", we will also visit the parent domains "leetcode.com" and "com" implicitly.

A count-paired domain is a domain that has one of the two formats "rep d1.d2.d3" or "rep d1.d2" where rep is the 
number of visits to the domain and d1.d2.d3 is the domain itself.

    For example, "9001 discuss.leetcode.com" is a count-paired domain that indicates that discuss.leetcode.com was 
    visited 9001 times.

Given an array of count-paired domains cpdomains, return an array of the count-paired domains of each subdomain in the 
input. You may return the answer in any order.

Input: cpdomains = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
Output: ["901 mail.com","50 yahoo.com","900 google.mail.com","5 wiki.org","5 org","1 intel.mail.com","951 com"]
Explanation: We will visit "google.mail.com" 900 times, "yahoo.com" 50 times, "intel.mail.com" once and "wiki.org" 
5 times.
For the subdomains, we will visit "mail.com" 900 + 1 = 901 times, "com" 900 + 50 + 1 = 951 times, and "org" 5 times.
"""


def website_domain(cpdomains):
    ans = dict()
    for domain in cpdomains:
        count, domain = domain.split()
        count = int(count)
        frags = domain.split('.')
        for i in range(len(frags)):
            key = ".".join(frags[i:])
            if key in ans.keys():
                ans[key] += count
            else:
                ans[key] = count

    list_ = []
    for key, value in ans.items():
        list_.append([value, key])

    return list_


cpdomains = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
output = website_domain(cpdomains)
print(output)

"""
A peak element is an element that is strictly greater than its neighbors.

Given an integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return 
the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞.

You must write an algorithm that runs in O(log n) time.

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the 
peak element is 6.
"""


def peak(arr):
    start = 0
    end = len(arr) - 1
    while start < end:
        mid_point = int((start + end) / 2)
        print(mid_point)
        if arr[mid_point] >= arr[mid_point - 1] and arr[mid_point] >= arr[mid_point + 1]:
            return arr[mid_point]
        if arr[mid_point] < arr[mid_point + 1]:
            start = mid_point + 1
        elif arr[mid_point] < arr[mid_point - 1]:
            end = mid_point - 1

    return -1


arr = [5, 0, -1, 0, 1, 2]
output = peak(arr)
print(output)

"""
Given a string s, return the longest palindromic substring in s.

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.

Example 2:

Input: s = "cbbd"
Output: "bb"
"""

"""
Given a signed 32-bit integer x, return x with its digits reversed. 
If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

Example 1:

Input: x = 123
Output: 321

Example 2:

Input: x = -123
Output: -321

Example 3:

Input: x = 120
Output: 21
"""


def reverse(integer_val):
    integer_val = list(str(integer_val))
    integer_val = integer_val[::-1]
    integer_val = "".join(integer_val)
    integer_val = int(integer_val)
    return integer_val


def reversed_image(input_matrix):
    num_rows = 3
    num_cols = 3

    output = [[0 for i in range(num_rows)] for j in range(num_cols)]
    print(output)

    for i in range(num_rows):
        for j in range(num_cols):
            output[i][j] = input_matrix[2 - j][i]

    return output


# input = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# output = reversed_image(input)
# print(output)

fruits = ['apple', 'banana', 'cherry']


def isPalindrome(s: str):
    s = [item for item in s.lower() if item.isalnum()]
    return s == s[::-1]


def longestCommonPrefix(strs):
    len_list = len(strs)

    items_size = [len(item) for item in strs]
    min_len = min(items_size)

    new_s = ""
    for j in range(min_len):
        count = 0
        for i in range(1, len_list):
            if strs[i - 1][j] == strs[i][j]:
                count += 1
            else:
                return new_s
        if count == len_list - 1:
            new_s += strs[0][j]


def longestCommonPrefix_1(strs):
    prefix = ""
    if len(strs) == 0:
        return prefix

    for j in range(len(min(strs))):
        c = strs[0][j]
        if all(a[j] == c for a in strs):
            prefix += c
        else:
            break
    return prefix


strs = ["hello", "hell"]
print(longestCommonPrefix_1(strs))


def maxDepth(self, root) -> int:
    print(root)

    if root is None:
        return 0
    else:
        left_height = self.maxDepth(root.left)
        right_height = self.maxDepth(root.right)
        return max(left_height, right_height) + 1


def Set_Matrix_Zeroes(matrix):
    cols = range(len(matrix))
    rows = range(len(matrix[0]))

    for i in rows:
        for j in cols:
            if matrix[i][j] == 0:
                matrix[i][:] = [0] * len(cols)
                matrix[:][j] = [0] * len(rows)

    return matrix


matrix = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
print(Set_Matrix_Zeroes(matrix))

word = "hello"

from collections import Counter


def anagrams(word1, word2):
    return Counter(word1) == Counter(word2)


list_strs = []

# def groupAnagrams(strs):
#     list_strs = []
#     for word1 in strs:
#         temp_group = [strs[i]]
#         for word2 in strs:
#             if anagrams(strs[i], strs[j]):
#                 temp_group.append(strs[j])
#         list_strs.append(temp_group)
#
#     return list_strs
#
#
# strs = ["eat","tea","tan","ate","nat","bat"]
# print(groupAnagrams(strs))


animals = ['cat', 'dog', 'dog', 'guinea pig', 'dog']

# 'dog' is removed
animals.remove('dog')

print(animals)

from collections import defaultdict


def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for value in coins:
        dp[value] = 1

    for value in coins:
        for invested in range(len(coins), amount + 1):
            dp[invested] = min(dp[invested - value] + 1, dp[invested])

    return dp[amount]


print(coinChange([1, 2, 4, 5], 11))

from collections import defaultdict


def canFinish(numCourses, prerequisites):
    courseDict = defaultdict(list)

    for relation in prerequisites:
        preCourse, nextCourse = relation[0], relation[1]
        courseDict[preCourse].append(nextCourse)

    print(courseDict)
    for key in courseDict.keys():
        if key in courseDict[key]:
            return False

    return True


print(canFinish(3, [[1, 2], [2, 3], [3, 4], [4, 1], [4, 2], [2, 1]]))


def productExceptSelf(nums):
    list_num = []
    for i in range(len(nums)):
        prod = 1
        count = 0
        for number in nums:
            if i != count:
                prod *= number
            count += 1

        list_num.append(prod)

    return list_num


# nums = [1, 2, 3, 4]
# print(productExceptSelf(nums))


def canPartition(nums):
    print("******")
    if sum(nums) % 2 == 1:
        return False

    partition_sum = sum(nums) / 2

    for partition in range(2 ** len(nums)):
        bin_num = bin(partition)[2:]

        bin_num_list = list(bin_num)
        num_int = [int(x) for x in bin_num_list]
        print(num_int)

        total_sum = 0
        for i in range(len(num_int)):
            if num_int[i] == 1:
                total_sum += nums[i]

        if total_sum == partition_sum:
            return True

    return False


# print(canPartition([2, 5, 6, 5, 6]))

from itertools import product


def topKFrequent(nums):
    from collections import Counter

    counter = Counter(nums)

    sorted_tuples = sorted(counter.items(), key=lambda item: item[1])
    sorted_dict = {k: v for k, v in sorted_tuples}

    list_ = []
    for key, value in sorted_dict.items():
        list_.append(key)

    return list_[-k:]


# Python3 Program to print BFS traversal
# from a given source vertex. BFS(int s)
# traverses vertices reachable from s.

# BFS algorithm
def bfs(graph, s):
    visited = [False] * (max(graph) + 1)

    # Create a queue for BFS
    queue = []

    # Mark the source node as
    # visited and enqueue it
    queue.append(s)
    visited[s] = True
    path = []

    while queue:

        # Dequeue a vertex from
        # queue and print it
        node = queue.pop(0)
        path.append(node)

        # Get all adjacent vertices of the
        # dequeued vertex s. If a adjacent
        # has not been visited, then mark it
        # visited and enqueue it
        for neighbor in graph[node]:
            if visited[neighbor] == False:
                queue.append(neighbor)
                visited[neighbor] = True

    return path


def bfs(graph, s):
    visited = [False] * len(graph)

    queue = []
    path = []

    queue.append(s)
    visited[s] = True
    path.append(s)

    while queue:
        node = queue.pop(0)
        path.append(node)

        for neighbor in graph[node]:
            if visited[neighbor] == False:
                visited[neighbor] = True
                queue.append(neighbor)


graph = {0: [1, 2], 1: [2], 2: [3], 3: [1, 2]}
print("Following is Breadth First Traversal: ")
print(bfs(graph, 0))

# array
# Given two strings ransomNote and magazine, return true if ransomNote can be constructed from magazine and false otherwise.
# Each letter in magazine can only be used once in ransomNote.

import numpy as np

arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

arr3 = np.dot(arr1, arr2)
print(arr3[0][0])


from collections import Counter

def twoSum(nums, target):
    # https://leetcode.com/problems/two-sum/
    hashmap = dict()
    list_twosum = []
    for i in range(len(nums)):
        hashmap[nums[i]] = i
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap.keys() and hashmap[complement] != i:
            list_twosum.append(sorted([i, hashmap[complement]]))

    list_twosum_ = []
    for item in list_twosum:
        if item not in list_twosum_:
            list_twosum_.append(item)

    return list_twosum_

arr = [1, 5, 4, 4, 7, 9]
output = twoSum(arr, 10)
print(output)

def valid_parentheses(s):
    # https://leetcode.com/problems/two-sum/

    open_par = []
    for symbol in s:
        if symbol in ["(", "[", "{"]:
            open_par.append(symbol)
        else:
            if (symbol == ")" and open_par[-1] == "(") or (symbol == "]" and open_par[-1] == "[") or (
                    symbol == "}" and open_par[-1] == "{"):
                open_par.pop()
            else:
                return False

    if len(open_par) == 0:
        return True


s = "()"
print(valid_parentheses(s))


def mergeTwoLists(list1, list2):
    # https://leetcode.com/problems/merge-two-sorted-lists/
    sort_list = []
    for _ in range(len(list1) + len(list2)):
        if len(list1) == 0:
            sort_list.append(list2[0])
            return sort_list

        if len(list2) == 0:
            sort_list.append(list1[0])
            return sort_list

        if list1[0] <= list2[0]:
            sort_list.append(list1[0])
            list1 = list1[1:]
        else:
            sort_list.append(list2[0])
            list2 = list2[1:]

    return sort_list


list1 = [1, 2, 4]
list2 = [1, 3, 4]
print(mergeTwoLists(list1, list2))


def best_time_to_buy_and_sell_stock(prices):
    # https: // leetcode.com / problems / best - time - to - buy - and -sell - stock
    min_price = float('inf')
    max_profit = 0
    for i in range(len(prices)):
        if prices[i] < min_price:
            min_price = prices[i]
        elif prices[i] - min_price > max_profit:
            max_profit = prices[i] - min_price

    return max_profit


prices = [8, 3, 5, 3, 6, 1]
print(best_time_to_buy_and_sell_stock(prices))


def isPalindrome(s):
    s = [item for item in s.lower() if item.isalnum()]
    return s == s[::-1]


def isAnagram(s, t):
    # https://leetcode.com/problems/valid-anagram/
    from collections import Counter

    counter_s = Counter(s)
    counter_t = Counter(t)

    if counter_s == counter_t:
        return True
    else:
        return False


s = "anagram"
t = "nagaram"
print(isAnagram(s, t))


def maxSubArray(nums):
    max_sum = 0
    new_sum = 0
    for num in nums:
        new_sum = num + new_sum

        if new_sum < 0:
            new_sum = 0

        if new_sum > max_sum:
            max_sum = new_sum

    return max_sum


def canConstruct(ransomNote, magazine):
    ransomNote_dict = Counter(ransomNote)
    magazine_dict = Counter(magazine)

    for key in ransomNote_dict.keys():
        if key in magazine_dict.keys():
            if ransomNote_dict[key] > magazine_dict[key]:
                return False
        else:
            return False

    return True


ransomNote = "tabacf"
magazine = "aacfdb"
print(canConstruct(ransomNote, magazine))


def longestPalindrome(s):
    s_dict = Counter(s)

    longest = 0
    count = 0
    for key in s_dict.keys():
        if s_dict[key] % 2 == 1:
            count += 1
            longest += 2 * (s_dict[key] // 2)
        else:
            longest += s_dict[key]

    if count > 0:
        longest += 1

    return longest


s = "abccceecdd"
print(longestPalindrome(s))


def majorityElement(nums):
    # https: // leetcode.com / problems / majority - element /
    nums_dict = Counter(nums)
    print(nums_dict)
    max_val = 0
    for key in nums_dict.keys():
        if nums_dict[key] >= max_val:
            max_key = key
            max_val = nums_dict[key]

    return max_key


nums = [2, 2, 1, 1, 1, 2, 2]
print(majorityElement(nums))


def canAttendMeetings(intervals):
    # https://leetcode.com/problems/meeting-rooms/
    intervals = sorted(intervals, key=lambda x: x[1])

    for index in range(len(intervals) - 1):
        if intervals[index][1] > intervals[index + 1][0]:
            return False

    return True


intervals = [[0, 30], [5, 10], [15, 20]]
print(canAttendMeetings(intervals))


def backspaceCompare(s, t):
    # https://leetcode.com/problems/backspace-string-compare/

    list_1 = []
    list_2 = []
    for letter in s:
        if letter != "#":
            list_1.append(letter)
        if letter == "#" and len(list_1) > 0:
            list_1.pop()

    for letter in t:
        if letter != "#":
            list_2.append(letter)
        if letter == "#" and len(list_2) > 0:
            list_2.pop()

    if list_1 == list_2:
        return True
    else:
        return False


s = "abc"
t = "ad#c"
print(backspaceCompare(s, t))


def longestCommonPrefix(strs):
    # https: // leetcode.com / problems / longest - common - prefix /
    len_list = len(strs)

    items_size = [len(item) for item in strs]
    min_len = min(items_size)

    new_s = ""
    for j in range(min_len):
        count = 0
        for i in range(1, len_list):
            if strs[i - 1][j] == strs[i][j]:
                count += 1
            else:
                return new_s
        if count == len_list - 1:
            new_s += strs[0][j]
    return new_s


def moveZeroes(nums):
    new_nums = nums.copy()
    count = 0
    for num in nums:
        if num == 0:
            print(num)
            count += 1
            new_nums.remove(0)

    new_nums.extend([0] * count)
    return new_nums


nums = [0, 1, 0, 3, 12]
print(moveZeroes(nums))

import math


def kClosest(points, k):
    def distance(x):
        return math.sqrt(x[0] * x[0] + x[1] * x[1])

    points_distance = list(map(distance, points))

    tuple_pairs = list(zip(points_distance, points))

    tuple_pairs = sorted(tuple_pairs)

    point_list = []
    for i in range(k):
        point_list.append(tuple_pairs[i][1])

    return point_list


points = [[1, 3], [-2, 2]]
k = 1
print(kClosest(points, k))


def lengthOfLongestSubstring(s):
    substring = ""
    max_sub = ""
    for ch in s:
        if ch in substring:
            index = substring.find(ch)
            substring = substring[index + 1:]

        substring += ch

        if len(substring) > len(max_sub):
            max_sub = substring

    return len(max_sub)


s = "pwweeeeerrrrtykew"
print(lengthOfLongestSubstring(s))


def coinChange(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] = min(dp[x], dp[x - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1


def alternatingCharacters(s):
    # Write your code here
    stack = []
    string = list(s)
    count = 0
    stack.append(string[0])
    for letter in string[1:]:
        stack.append(letter)
        if stack[-1] == stack[-2]:
            stack.pop()
            print(stack)

    return len(string) - len(stack)


s = "ABABABAB"
print(alternatingCharacters(s))

# ----------------------------------------------------------------------------------------------------------------------
import collections

0
def longestConsecutive(nums):
    # https: // leetcode.com / problems / longest - consecutive - sequence /
    nums = sorted(nums)

    count = 0
    ans = 0
    for num_index in range(len(nums) - 1):
        if nums[num_index] == nums[num_index + 1]:
            count = +1
        else:
            ans = max(ans, count)
            count = 0

    return ans


def soduko_


nums = [1, 2, 4, 3, 5, 6, 7, 9, 12]
dict_nums = longestConsecutive(nums)
print(dict_nums)
