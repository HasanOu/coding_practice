import collections
import numpy as np

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
