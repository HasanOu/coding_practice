import collections
import numpy as np
from collections import Counter
from collections import defaultdict
from typing import List

"""Given an array of integers nums and an integer target, return indices of the
two numbers such that they add up to target.

You may assume that each input would have exactly one solution,
and you may not use the same element twice.

You can return the answer in any order.

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""

# =========================== Two Sum =============================
def twoSum(nums, target):
    # create a dictionary to store each number and its index
    complement_dict = {}

    # loop through each number in the array
    for i, num in enumerate(nums):

        # calculate the complement (the number we need to reach the target)
        complement = target - num

        # if the complement is already in the dictionary, we found a solution
        if complement in complement_dict.keys():
            return [complement_dict[complement], i]

        # if the complement is not in the dictionary, add the current number and its index to the dictionary
        complement_dict[num] = i

    # if no solution was found, return an empty array
    return []

arr = [1, 5, 4, 4, 7, 9]
output = twoSum(arr, 10)
print(output)
print("***********")

# ============================= Valid Parentheses ============================
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

my_string = "(({}))"
print(IsValid(my_string))
print("*******")

# ========================= Best Time to Buy and Sell Stock ========================
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
    end = len(my_list) - 1

    max_profit = max(0, my_list[end] - my_list[start])

    if my_list[start] > my_list[start + 1]:
        start += 1
        max_profit = my_list[end] - my_list[start]

    if my_list[end] < my_list[end - 1]:
        end -= 1
        max_profit = my_list[end] - my_list[start]

    start += 1
    end -= 1
    max_profit = max(max_profit, my_list[end] - my_list[start])

    return max_profit

prices = [7, 1, 15, 3, 16, 4]
print(buy_sell_profit(prices))
print("*******")

# ================================= Valid Palindrome ==============================
"""A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.
Given a string s, return true if it is a palindrome, or false otherwise."""

def Validpalindrome(my_string):
    my_string = my_string.replace(" ", "").lower()

    if my_string[::-1] == my_string:
        return True
    else:
        return False

def Validpalindrome2(my_string):
    my_string = my_string.replace(" ", "").lower()

    left, right = 0, len(s)-1
    while left < right:
        if my_string[left]!= my_string[right]:
            return False
        left+=1
        right-=1
    return True

# ================================ Valid Anagram ==========================
def anagrams(word1, word2):
    return Counter(word1) == Counter(word2)

def anagrams2(word1, word2):
    word1_freq = {}

    if len(word1) != len(word2):
        return False

    for char in word1:
        word1_freq[s]= word1_freq.get(char,0)+1

    for char in word2:
        if char not in word1_freq.keys():
            return False
        word1_freq[char] -= 1
        if word1_freq[char] < 0:
            return False
    return True

# ================================ Binary Search ========================
"""
Given an array of integers nums which is sorted in ascending order, 
and an integer target, write a function to search target in nums. If target exists, then return its index. 
Otherwise, return -1.
You must write an algorithm with O(log n) runtime complexity.

nums = [-1,0,3,5,9,12], target = 9
output: 4
"""

def binary_search(nums, target):
    left = 0
    right = len(nums)-1

    while left <= right:
        mid = int((left+right)/2)

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid+1
        else:
            right = mid-1
    return -1

nums = [-1,0,3,5,9,12]
target = 30
print(binary_search(nums,target))
print("***********")

"""
An image is represented by an m x n integer grid image where image[i][j] 
represents the pixel value of the image.

You are also given three integers sr, sc, and color. 
You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, 
plus any pixels connected 4-directionally to the starting pixel of the same color 
as the starting pixel, plus any pixels connected 4-directionally to those pixels 
(also with the same color), and so on. 
Replace the color of all of the aforementioned pixels with color.

Return the modified image after performing the flood fill.
"""

def floodFill(image, sr, sc, newColor):
    oldColor = image[sr][sc]

    if oldColor == newColor:
        return image

    def dfs(image, r, c, oldColor, newColor):
        num_rows = len(image)
        num_cols = len(image[0])

        if (r < 0 or r >= num_rows) or (c < 0 or c >= num_cols) or image[r][c] != oldColor:
            return

        image[r][c] = newColor

        dfs(image, r + 1, c, oldColor, newColor)
        dfs(image, r - 1, c, oldColor, newColor)
        dfs(image, r, c + 1, oldColor, newColor)
        dfs(image, r, c - 1, oldColor, newColor)

    dfs(image, sr, sc, oldColor, newColor)
    return image

image = [[3,1,1],[1,1,0],[1,0,1]]
sr = 1
sc = 1
color = 2
print(floodFill(image,1,1,2))
print("***********")

# ================================== First bad version ========================
"""You are a product manager and currently leading a team to develop a new product. 
Unfortunately, the latest version of your product fails the quality check. 
Since each version is developed based on the previous version, 
all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, 
which causes all the following  ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.
"""

def firstBadVersion(n):
    # Initialize the search range
    left = 1
    right = n

    # Perform binary search
    while left < right:
        # Calculate the mid-point
        mid = int((left+right)/2)

        # Check if the mid-point is a bad version
        if isBadVersion(mid):
            right = mid
        else:
            left = mid + 1

    # Return the first bad version
    return left

 # ================================  Climbing Stairs =========================
"""
You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
"""

def climbStairs(n):
    # Initialize the memoization table
    dp = [0] * (n + 1)

    # Base cases
    dp[0] = 0
    dp[1] = 1
    dp[2] = 2

    # Fill in the memoization table using dynamic programming
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]

    # Return the number of distinct ways to climb the staircase
    return dp[n]


# ================================= Longest Palindrome =======================
"""
Given a string s which consists of lowercase or uppercase letters, 
return the length of the longest palindrome that can be built with those letters.

Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

Input: s = "abccccdd"
Output: 7
Explanation: One longest palindrome that can be built is "dccaccd", whose length is 7.
"""

def Langestpalindrome(my_string):
    dict = Counter(my_string)

    max_length = 0
    mid_val = 0
    for (key, value) in dict.items():
        if value % 2 == 0:
            max_length += value
        else:
            max_length += value - 1
            mid_val = 1

    return max_length + mid_val

my_string = "dccaaaccd"
print(Langestpalindrome(my_string))
print("*******")


# ======================== Majority elements =============================
"""Given an array nums of size n, return the majority element. The majority element is 
the element that appears more than n/2 times. You may assume that the majority element always 
exists in the array."""

def MajorityElement(nums):
    dict = defaultdict(int)

    for num in nums:
        dict[num] += 1

    return max(dict.keys())

print(MajorityElement([2, 2, 1, 1, 1, 2, 2]))
print("***********")


# ================================ contains duplicate =============================
"""Given an integer array nums, return true if any value appears at least twice 
in the array, and return false if every element is distinct."""
def ContainsDuplicate(nums):
    duplciate_dict = defaultdict(int)

    for num in nums:
        duplciate_dict[num]=1
        if num in duplciate_dict.keys():
            return False
    return True

# ============================ Meeting room =======================================
"""Given an array of meeting time intervals intervals where 
intervals[i] = [start_i, end_i], determine if a person could attend all meetings."""
def MeetingRoom(intervals):
    sorted_list = sorted(intervals, key=lambda x: x[0])
    pair_list = list(zip(intervals, sorted_list[1:]))

    for (first, second) in pair_list:
        if first[1] > second[0]:
            return False

    return True

intervals = [[0, 5], [5, 10], [10, 15]]
print(MeetingRoom(intervals))
print("*********")


# ============================== backspace string compare =====================
"""
Given two strings s and t, return true if they are equal when both are typed 
into empty text editors. '#' means a backspace character.

Note that after backspacing an empty text, the text will continue empty.

Input: s = "ab#c", t = "ad#c"
Output: true
Explanation: Both s and t become "ac".
"""

def Backspace_String_Compare(s, t):
    new_s = []
    for item in s:
        new_s.append(item)
        if item == "#":
            new_s = new_s[:-2]

    new_t = []
    for item in t:
        new_t.append(item)
        if item == "#":
            new_t = new_t[:-2]

    if new_s == new_t:
        return True
    else:
        return False

s = "ab##"
t = "c#d#"
print(Backspace_String_Compare(s, t))
print("*********")

# =========================== Longest common prefix ==========================
"""
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".
"""

def longest_common_prefix(strs):
    min_size = min(list(map(len, strs)))
    min_size = min([len(s) for s in strs])
    count = 0
    total_count = 0
    for char in range(min_size):
        for index in range(len(strs) - 1):
            if strs[index][char] == strs[index + 1][char]:
                count = max(1, count)
            else:
                count = 0
        total_count += count
    return strs[0][:total_count]


strs = ["flower", "flow", "flight"]
print(longest_common_prefix(strs))
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


list1 = [1, 2, 4]
list2 = [2, 3, 5, 8]
print(MergeLists(list1, list2))
print("*****")






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



"""
Given two strings s and t, return true if t is an anagram of s, and false otherwise.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

"abcabc" and "cabcab"

"""


def Validanagrams(word1, word2):
    return Counter(word1) == Counter(word2)


import copy


def Validanagram(word1, word2):
    word1 = list(word1)
    word2 = list(word2)

    my_iter = copy.deepcopy(word1)

    if len(word1) != len(word2):
        return False
    else:
        for char in my_iter:
            if char in word2:
                word1.remove(char)
                word2.remove(char)
            else:
                return False

    if len(word1) == 0 and len(word2) == 0:
        return True
    else:
        return False


word1 = "aabc"
word2 = "caab"
print(Validanagram(word1, word2))
print("**************")

# Alternative solution
def backspaceCompare(s, t):
    # Define a function to simulate backspacing in a string
    def backspace(s):
        stack = []
        for char in s:
            if char == '#':
                if stack:
                    stack.pop()
            else:
                stack.append(char)
        return stack

    # Use the backspace function to compare the modified strings
    return backspace(s) == backspace(t)


# 10-10:15
"""
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.

You must implement a solution with a linear runtime complexity and use only constant extra space.
"""


def singleNumber(nums):
    my_dict = defaultdict(int)

    for num in nums:
        my_dict[num] += 1

    for (key, value) in my_dict.items():
        if value % 2 == 1:
            return key


nums = [4, 1, 2, 1, 2]
print(singleNumber(nums))
print("********")

# 10:15-10:30
"""Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array."""


def move_zeros(nums):
    for num in nums:
        if num == 0:
            nums.append(0)
            nums.remove(0)
    return nums


nums = [0, 0, 1, 0, 2, 0, 3]
print(move_zeros(nums))
print("**********")

# 10:30-10:45
"""
Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.
"""


def missing_num(my_list):
    my_dict = defaultdict(int)
    for num in nums:
        my_dict[num] = 1

    for index in range(len(my_list)):
        if my_dict[index] == 0:
            return index


nums = [9, 6, 4, 2, 3, 5, 7, 0, 1]
print(missing_num(nums))
print("*********")

# 10:45-11
"""Given an integer x, return true if x is a
palindrome , and false otherwise."""


def Palindrome_num(num):
    if num < 0:
        return False

    num = list(str(num))
    start = 0
    end = len(num) - 1

    while start < end:
        if num[start] == num[end]:
            start += 1
            end -= 1
        else:
            return False
    return True


num = 121
print(Palindrome_num(num))
print("********")

# 11-11:15
"""
Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.
Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
"""


def sortedSquares(nums):
    positive_list = [x ** 2 for x in nums if x >= 0]
    negative_list = [x ** 2 for x in nums if x < 0][::-1]

    # [4, 9, 12]
    # [9, 49]

    list_size = len(nums)
    new_list = []

    for i in range(list_size):
        if len(positive_list) == 0:
            new_list += negative_list
            return new_list

        if len(negative_list) == 0:
            new_list += positive_list
            return new_list

        if positive_list[0] < negative_list[0]:
            new_list.append(positive_list[0])  # [4] [9]
            positive_list = positive_list[1:]  # [12]
        else:
            new_list.append(negative_list[0])  # [9]
            negative_list = negative_list[1:]  # [49]

    return new_list


nums = [-7, -3, 2, 3, 11]
print(sortedSquares(nums))
print("*******")

# 11:15-11:30
"""
Given an integer array nums, find the subarray with the largest sum, and return its sum.
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
"""


def MaximumSubarray(nums):
    temp_sum = 0
    max_sum = 0
    for num in nums:
        temp_sum += num
        if temp_sum < 0:
            temp_sum = 0
        else:
            max_sum = max(max_sum, temp_sum)
    return max_sum


nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
print(MaximumSubarray(nums))
print("*********")

# 11:30-11:45
"""
You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]] 
"""


def InsertInterval(intervals, new_interval):
    res = []
    i = 0

    while i < len(intervals) and intervals[i][1] < new_interval[0]:
        res.append(intervals[i])
        i += 1

    while i < len(intervals) and intervals[i][1] > new_interval[0] and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1

    res.append(new_interval)

    while i < len(intervals):
        res.append(intervals[i])
        i += 1

    return res


intervals = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]
newInterval = [4, 8]
print(InsertInterval(intervals, newInterval))
print("**********")

"""
Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.
"""

"""
Given a string s, find the length of the longest
substring without repeating characters.
"""


def longestsubstring(my_str):
    my_list = list(my_str)
    max_size = 0
    new_list = []
    i = 0
    while i < len(my_list):
        char = my_list[i]
        i += 1
        if char not in new_list:
            new_list.append(char)

        else:
            max_size = max(max_size, len(new_list))
            new_list = []
            index = my_list.index(char)
            i = index + 1

    return max(max_size, len(new_list))


my_str = "abcddfgtrtyuio"
print(longestsubstring(my_str))
print("*******")

"""
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
nums = [-1,0,1,2,-1,-4]
"""


def threeSum(nums):
    """
    Given an array nums of n integers, find all unique triplets in the array
    which gives the sum of zero. The solution set must not contain duplicate triplets.

    Args:
        nums (List[int]): The input list of integers.

    Returns:
        List[List[int]]: The list of unique triplets that sum to zero.
    """
    nums.sort()
    result = []

    for i in range(len(nums) - 2):

        left, right = i + 1, len(nums) - 1

        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                left += 1
                right -= 1

    new_result = []
    for item in result:
        if item not in new_result:
            new_result.append(item)

    return new_result


nums = [-1, 0, 1, 2, -1, -4]
print(threeSum(nums))
print("**********")

"""
Evaluate Reverse Polish Notation
You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation.
Evaluate the expression. Return an integer that represents the value of the expression.
"""


def ReversePolishNotation():
    pass


"""
Course schedule
There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates 
that you must take course bi first if you want to take course ai.

    For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.

Return true if you can finish all courses. Otherwise, return false. 
"""

"""You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.
coins = [1,2,5], amount = 11
"""

def coin_change(coins: List[int], amount: int) -> int:
    dp = [0] * (amount + 1)

    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    dp[3] = 2
    dp[4] = 2
    dp[5] = 1

    for val in range(6, amount + 1):
        dp[val] = min(dp[val - 1], dp[val - 2], dp[val - 5]) + 1
    return dp[amount]

# alternative solution

def gpt_coin_change(coins: List[int], amount: int) -> int:
    # Create a table to store the fewest number of coins needed for each sub-amount
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    # Iterate through each coin and each sub-amount, updating the fewest number of coins
    # needed to make up that amount if a better solution is found.
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)

coins = [1, 2, 5]
amount = 24
print(coin_change(coins, amount))
print("**********")

"""
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

You must write an algorithm that runs in O(n) time and without using the division operation.
"""

def productExceptSelf(nums):
    new_list = []
    for index in range(len(nums)):
        left = 1
        right = 1

        start = 0
        while start < index:
            left = left*nums[start]
            start+=1

        end = len(nums) - 1
        while end > index:
            right = right*nums[end]
            end-=1

        new_list.append(left*right)

    return new_list

nums = [1,2,3,4]
print(productExceptSelf(nums))
print("*********")

"""
There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) 
such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). 
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.
"""

def rotatedarraysearch(nums, target):
    left = 0
    right = len(nums)-1

    mid = (left+right)//2

    if nums[mid] == target:
        return mid

    while left <= right:
        mid = (left+right)//2

        if nums[mid] == target:
            return mid

        # the left side is sorted and the target is in the left side
        if nums[left] <= nums[mid]:
            if nums[left] <= target <= nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] <= target <= nums[right]:
                left = mid + 1
            else:
                right = mid -1
    return -1

nums = [4,5,6,7,0,1,2]
target = 0
print(rotatedarraysearch(nums, target))
print("********")

"""
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:

    void push(int x) Pushes element x to the back of the queue.
    int pop() Removes the element from the front of the queue and returns it.
    int peek() Returns the element at the front of the queue.
    boolean empty() Returns true if the queue is empty, false otherwise.
    
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
"""


class MyQueue:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        self.stack1.append(x)

    def pop(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()

    def peek(self):
        if not self.stack2:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
        return self.stack2[-1]

    def empty(self):
        return not self.stack1 and not self.stack2


queue = MyQueue()
queue.push(1)
queue.push(2)
print(queue.peek())  # Output: 1
print(queue.pop())  # Output: 1
print(queue.empty())

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


nums = [1, 2, 4, 3, 5, 6, 7, 9, 12]
dict_nums = longestConsecutive(nums)
print(dict_nums)

# 10-10:30
# ========================== Combination Sum =========================================

"""
Given an array of distinct integers candidates and a target integer target, 
return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the
frequency
of at least one of the chosen numbers is different.
The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input."""

def combinationsum(candidates, target):
    def backtrack(target, path):
        if target < 0:
            return
        if target ==0:
            res.append(path)
            return
        for num in candidates:
            backtrack(target-num, path+[num])

    res =[]
    candidates.sort()
    backtrack(target, [])
    return res

candidates = [2,3,6,7]
target = 7
print(combinationsum(candidates, target))
print("******")

# """
# Given an array nums of distinct integers, return all the possible permutations.
# You can return the answer in any order.
# """
#
# # my solution but it is wrong
# def permutation(nums):
#     def construct_permutation(num, item):
#         new_per_list = []
#         for i in range(len(item) + 1):
#             new_perm = item[:i] + [num] + item[i:]
#             new_per_list.append(new_perm)
#
#         return new_per_list
#
#     def backtrack(num, permut):
#         # for each item in permutation, create the permutation of that item
#         current_permut = []
#         for item in permut:
#            for index in range(len(item)):
#                new_permut = construct_permutation(num, item)
#                current_permut.extend(new_permut)
#
#         result = current_permut
#         print("******")
#         permut = result
#
#         for num in nums[2:]:
#            backtrack(num, result)
#
#     permut_list = [[nums[1]]]
#     result = []
#     backtrack(num = nums[0], permut=permut_list)
#     return result
#
# nums = [1,2,3]
# print(permutation(nums))
# print("******")
#
# # ==========================
# # chatgpt solution
# def permutation(nums):
#     def construct_permutation(num, item):
#         new_per_list = []
#         for i in range(len(item) + 1):
#             new_perm = item[:i] + [num] + item[i:]
#             new_per_list.append(new_perm)
#
#         return new_per_list
#
#     def backtrack(nums, permut):
#         if not nums:
#             result.append(permut)
#         else:
#             for i in range(len(nums)):
#                 new_per = construct_permutation(nums[i], permut)
#                 backtrack(nums[:i] + nums[i+1:], new_per)
#
#     result = []
#     backtrack(nums, [])
#     return result
#
# nums = [1,2,3]
# print(permutation(nums))


def fib_func(n):

    fib = [0]*(n+1)
    fib[0] = 1
    fib[1] = 1

    for n in range(2,n):
        fib[n] = fib[n-1]+fib[n-2]

    return fib[n]

print(fib_func(10))


"""
Given a list of accounts where each element accounts[i] is a list of strings, 
where the first element accounts[i][0] is a name, and the rest of the elements are emails 
representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. 
A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. 
The accounts themselves can be returned in any orde
"""

from collections import defaultdict

def merge_accounts(accounts):
    # Create a defaultdict to store emails and corresponding names
    email_to_name = defaultdict(str)

    # Create a defaultdict to store emails and corresponding emails
    graph = defaultdict(set)

    # Iterate through each account and their emails
    for account in accounts:
        name = account[0]
        emails = account[1:]

        # Add the email to name mapping
        for email in emails:
            email_to_name[email] = name

        # Build the graph for each email in the account
        for i in range(len(emails)):
            for j in range(i + 1, len(emails)):
                graph[emails[i]].add(emails[j])
                graph[emails[j]].add(emails[i])

    # Create a list to store the merged accounts
    merged_accounts = []

    # Iterate through each email and their corresponding name

    # Create a set to keep track of visited emails
    visited = set()

    for email, name in email_to_name.items():
        if email not in visited:
            visited.add(email)

            # Create a set to store all emails in the same group
            group = {email}

            # Perform BFS on the graph to find all connected emails
            queue = [email]
            while queue:
                curr_email = queue.pop(0)
                for neighbor in graph[curr_email]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        group.add(neighbor)
                        queue.append(neighbor)

            # Add the merged account to the result
            merged_account = [name] + sorted(list(group))
            merged_accounts.append(merged_account)

    return merged_accounts

accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
print(merge_accounts(accounts))

# BFS implementation
from collections import deque

def bfs(graph, start_node):
    visited = set()
    queue = deque([start_node])

    while queue:
        curr_node = queue.popleft()
        visited.add(curr_node)

        # Do something with the current node, e.g. print it
        print(curr_node)

        for neighbor in graph[curr_node]:
            if neighbor not in visited:
                queue.append(neighbor)

graph = {1: [2,3], 2:[1,4],3:[1, 4,5],4:[3,6], 5:[3], 6: [4]}
node = 1

bfs(graph, 1)

def my_bfs(graph, start_node):
    visited = {}
    queue = deque([start_node])

    while queue:
        current_node = queue.popleft()
        visited.add(current_node)
        print(current_node)

    for neighbor in graph[current_node]:
        if neighbor not in visited:
            queue.add(neighbor)

## # DFS algo

def DFS(graph, start):
    visited = set()
    stack = [start]

    while stack:
        curr = stack.pop()

        if curr not in visited:
            visited.add(curr)

            # Do something with the current node
            print(curr)

            for neighbor in graph[curr]:
                if neighbor not in visited:
                    stack.append(neighbor)

# Example usage
graph = {
    1: [2, 3],
    2: [4, 5],
    3: [6, 7],
    4: [],
    5: [],
    6: [],
    7: []
}

DFS(graph, 1)

"""
Given an array nums with n objects colored red, white, or blue, 
sort them in-place so that objects of the same color are adjacent, 
with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to 
represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.
"""

def subset_sum(nodes, target):
    n = len(nodes)
    if target == 0:
        return []
    if n == 0 or target < nodes[0]:
        return None
    # include the first node
    include = subset_sum(nodes[1:], target - nodes[0])
    if include is not None:
        return [nodes[0]] + include
    else:
    # exclude the first node
        exclude = subset_sum(nodes[1:], target)
        return exclude

# Example usage
nodes = [1, 2, 3, 4, 5]
target = 8
result = subset_sum(nodes, target)
if result is None:
    print("No subset found")
else:
    print("Subset with sum", target, "found:", result)


"""Given an integer array nums of unique elements, return all possible
subsets (the power set). The solution set must not contain duplicate subsets. 
Return the solution in any order."""

def subset_func(nums):
    nums = sorted(nums)
    if not nums:
        return [[]]

    subset_list = subset_func(nums[1:])

    return subset_list + [[nums[0]] + subset for subset in subset_list]

nums = [1,2,3,4]
print(subset_func(nums))

"Given a string s, return the longest palindromic substring in s."

def longest_palindrome(s):
    substrings = []
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            substrings.append(s[i:j])
    palindromes = [sub for sub in substrings if sub == sub[::-1]]
    if palindromes:
        return max(palindromes, key=len)

s = 'abccbaf'
print(longest_palindrome(s))
