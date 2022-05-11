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


# arr = [1, 5, 4, 4, 7, 9]
# output = twoSum(arr, 10)
# print(output)


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
    for i in np.arange(len(arr)):
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
        for j in np.arange(i + 1, len(arr)):
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


def next_permutation(arr):
    if sorted(arr, reverse = True) == arr:
        return sorted(arr)

    hi = len(arr) - 1
    while arr[hi - 1] > arr[hi]:
        hi -= 1

    temp = arr[hi - 1]
    arr[hi - 1] = arr[hi]
    arr[hi] = temp

    return arr


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


def search_in_rotated_rorted_array(nums, target):
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
output = search_in_rotated_rorted_array(arr, 7)
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
    arr.sort(key = lambda x: x[0])

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
        mid_point = int((start+end)/2)
        print(mid_point)
        if arr[mid_point] >= arr[mid_point-1] and arr[mid_point] >= arr[mid_point+1]:
            return arr[mid_point]
        if arr[mid_point] < arr[mid_point+1]:
            start = mid_point + 1
        elif arr[mid_point] < arr[mid_point-1]:
            end = mid_point - 1

    return -1


arr = [5, 0, -1, 0, 1, 2]
output = peak(arr)
print(output)


"""
Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:

Input: nums = [3,2,1,5,6,4], k = 2
Output: 5

Example 2:

Input: nums = [3,2,3,1,2,4,5,5,6], k = 4
Output: 4

"""



