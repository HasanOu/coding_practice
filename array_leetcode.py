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
    for i in range(len(nums)):
        hashmap[nums[i]] = i
    for i in range(len(nums)):
        complement = target - nums[i]
        if complement in hashmap.keys():
            return [i, hashmap[complement]]

# arr = [1, 5,6, 7, 9]
# output = twoSum(arr, 11)
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
        max_area = max(max_area, width*min(arr[right], arr[left]))

        if arr[left] <= arr[right]:
            left += 1
        else:
            right -=1

    return max_area


# arr = [1, 8, 6, 2, 5, 4, 8, 3, 7]
# output = container_with_most_water(arr)
# print(output)

