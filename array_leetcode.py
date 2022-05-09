"""Given an array of integers nums and an integer target, return indices of the
two numbers such that they add up to target.

You may assume that each input would have exactly one solution,
and you may not use the same element twice.

You can return the answer in any order.

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].
"""

import numpy as np

def twoSum(nums, target):
    hashmap = dict()
    list_twosum = []
    for i in range(len(nums)):
        hashmap[nums[i]] = i
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap.keys() and hashmap[complement]!= i:
            list_twosum.append(sorted([i, hashmap[complement]]))

    list_twosum_ = []
    for item in list_twosum:
        if item not in list_twosum_:
            list_twosum_.append(item)

    return list_twosum_


#arr = [1, 5, 4, 4, 7, 9]
#output = twoSum(arr, 10)
#print(output)


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
        max_area = max(max_area, width*min(arr[right], arr[left]))

        if arr[left] <= arr[right]:
            left += 1
        else:
            right -=1

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


def threesum(arr,target):
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

#arr = [3,3,1,2,-1,4]
#output = threesum(arr, 6)
#print(output)


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
        lo, hi = i+1, len(arr) - 1

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

#arr = [-1,4,1,-4]
#output = threesum_closest(arr, 1)
#print(output)

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
        for j in np.arange(i+1, len(arr)):
            lo, hi = j+1, len(arr)-1

            while lo < hi:
                sum =  arr[i] + arr[j] + arr[lo] + arr[hi]
                if sum == target:
                    list_values.append([arr[i], arr[j], arr[lo], arr[hi]])

                if sum <= target:
                    lo+=1
                else:
                    hi-=1

    return list_values

output = foursum([1,0,-1,1,-2,2], 0)
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

arr = [0,0,1,1,1,2,2,3,3,4]
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
    if sorted(arr, reverse=True) == arr:
        return sorted(arr)

    hi = len(arr) -1
    while arr[hi-1] > arr[hi]:
        hi -= 1

    temp = arr[hi-1]
    arr[hi-1] = arr[hi]
    arr[hi] = temp

    return arr

arr= [2,3,1,4]
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


def search_in_rotated_rorted_array(arr, target):

    lo = 0
    hi = len(arr) - 1

    while lo < hi:
        middle_val = (arr[lo]+arr[hi])/2
        middle_point = int((lo+hi)/2)

        if target <= middle_val:
            hi = middle_point
        else:
            lo = middle_point

        if arr[middle_point] == target:
            return middle_point

    return -1

arr = [8, 0, 1,2,3,4]
output = search_in_rotated_rorted_array(arr, 8)
print(output)

