"""
Coding benchmark tasks and hidden tests.

Each task has:
- id: stable task name
- prompt: function specification sent to the model
- tests: Python assertions executed against solution.py
"""

TASKS = [
    {
        "id": "two_sum",
        "prompt": """
Implement:

def two_sum(nums: list[int], target: int) -> list[int]

Return the indices of two distinct elements whose sum is target.
There is exactly one valid answer. Return the indices in any order.
""",
        "tests": """
from solution import two_sum

r = two_sum([2, 7, 11, 15], 9)
assert set(r) == {0, 1}

r = two_sum([3, 2, 4], 6)
assert set(r) == {1, 2}

r = two_sum([-3, 4, 3, 90], 0)
assert set(r) == {0, 2}
""",
    },
    {
        "id": "longest_valid_parentheses",
        "prompt": """
Implement:

def longest_valid_parentheses(s: str) -> int

Return the length of the longest valid parentheses substring.
""",
        "tests": """
from solution import longest_valid_parentheses

assert longest_valid_parentheses("(()") == 2
assert longest_valid_parentheses(")()())") == 4
assert longest_valid_parentheses("") == 0
assert longest_valid_parentheses("()(())") == 6
""",
    },
    {
        "id": "coin_change",
        "prompt": """
Implement:

def coin_change(coins: list[int], amount: int) -> int

Return the minimum number of coins needed to make amount.
If impossible, return -1.
You may use each coin denomination unlimited times.
""",
        "tests": """
from solution import coin_change

assert coin_change([1, 2, 5], 11) == 3
assert coin_change([2], 3) == -1
assert coin_change([1], 0) == 0
assert coin_change([2, 5, 10, 1], 27) == 4
""",
    },
    {
        "id": "num_islands",
        "prompt": """
Implement:

def num_islands(grid: list[list[str]]) -> int

The grid contains "1" for land and "0" for water.
Return the number of 4-directionally connected islands.
""",
        "tests": """
from solution import num_islands
from copy import deepcopy

g1 = [
    ["1","1","1","1","0"],
    ["1","1","0","1","0"],
    ["1","1","0","0","0"],
    ["0","0","0","0","0"],
]
assert num_islands(deepcopy(g1)) == 1

g2 = [
    ["1","1","0","0","0"],
    ["1","1","0","0","0"],
    ["0","0","1","0","0"],
    ["0","0","0","1","1"],
]
assert num_islands(deepcopy(g2)) == 3

assert num_islands([]) == 0
""",
    },
    {
        "id": "edit_distance",
        "prompt": """
Implement:

def edit_distance(a: str, b: str) -> int

Return the Levenshtein edit distance between a and b.
Allowed operations: insert, delete, replace.
Each operation costs 1.
""",
        "tests": """
from solution import edit_distance

assert edit_distance("horse", "ros") == 3
assert edit_distance("intention", "execution") == 5
assert edit_distance("", "abc") == 3
assert edit_distance("abc", "abc") == 0
""",
    },
    {
        "id": "word_break",
        "prompt": """
Implement:

def word_break(s: str, word_dict: list[str]) -> bool

Return True if s can be segmented into one or more dictionary words.
Words may be reused.
""",
        "tests": """
from solution import word_break

assert word_break("leetcode", ["leet", "code"]) is True
assert word_break("applepenapple", ["apple", "pen"]) is True
assert word_break("catsandog", ["cats", "dog", "sand", "and", "cat"]) is False
assert word_break("", ["a"]) is True
""",
    },
    {
        "id": "topological_order",
        "prompt": """
Implement:

def topological_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]

There are courses 0..num_courses-1.
Each pair [course, prerequisite] means prerequisite must be taken before course.
Return any valid topological order.
If impossible because of a cycle, return [].
""",
        "tests": """
from solution import topological_order

def valid(n, prereq, order):
    if len(order) != n:
        return False
    if set(order) != set(range(n)):
        return False
    pos = {x: i for i, x in enumerate(order)}
    return all(pos[p] < pos[c] for c, p in prereq)

p1 = [[1, 0], [2, 0], [3, 1], [3, 2]]
assert valid(4, p1, topological_order(4, p1))

p2 = [[0, 1], [1, 0]]
assert topological_order(2, p2) == []

assert valid(1, [], topological_order(1, []))
""",
    },
    {
        "id": "shortest_path_binary_matrix",
        "prompt": """
Implement:

def shortest_path_binary_matrix(grid: list[list[int]]) -> int

You start at top-left and need to reach bottom-right.
0 means free cell, 1 means blocked cell.
You may move in 8 directions.
Return the length of the shortest clear path, or -1 if impossible.
""",
        "tests": """
from solution import shortest_path_binary_matrix
from copy import deepcopy

assert shortest_path_binary_matrix(deepcopy([[0, 1], [1, 0]])) == 2
assert shortest_path_binary_matrix(deepcopy([[0,0,0],[1,1,0],[1,1,0]])) == 4
assert shortest_path_binary_matrix(deepcopy([[1, 0], [0, 0]])) == -1
assert shortest_path_binary_matrix(deepcopy([[0]])) == 1
""",
    },
    {
        "id": "merge_intervals",
        "prompt": """
Implement:

def merge_intervals(intervals: list[list[int]]) -> list[list[int]]

Merge all overlapping intervals.
Return intervals sorted by start.
""",
        "tests": """
from solution import merge_intervals

assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([]) == []
assert merge_intervals([[1,4],[0,2],[3,5]]) == [[0,5]]
""",
    },
    {
        "id": "min_cost_cut_stick",
        "prompt": """
Implement:

def min_cost_cut_stick(n: int, cuts: list[int]) -> int

A stick has length n. You must perform all cuts.
The cost of each cut is the current length of the stick segment being cut.
Return the minimum total cost.
""",
        "tests": """
from solution import min_cost_cut_stick

assert min_cost_cut_stick(7, [1, 3, 4, 5]) == 16
assert min_cost_cut_stick(9, [5, 6, 1, 4, 2]) == 22
assert min_cost_cut_stick(10, []) == 0
""",
    },
    {
        "id": "dijkstra",
        "prompt": """
Implement:

def dijkstra(n: int, edges: list[tuple[int, int, int]], source: int) -> list[float]

The graph is directed and weighted with non-negative weights.
edges contains tuples (u, v, w).
Return a list dist where dist[i] is the shortest distance from source to i.
For unreachable nodes, use float("inf").
""",
        "tests": """
from solution import dijkstra

edges = [
    (0, 1, 2),
    (0, 2, 5),
    (1, 2, 1),
    (1, 3, 2),
    (2, 3, 1),
]
dist = dijkstra(5, edges, 0)
assert dist[0] == 0
assert dist[1] == 2
assert dist[2] == 3
assert dist[3] == 4
assert dist[4] == float("inf")

assert dijkstra(1, [], 0) == [0]
""",
    },
    {
        "id": "max_profit_k_transactions",
        "prompt": """
Implement:

def max_profit_k_transactions(prices: list[int], k: int) -> int

Given daily stock prices, return the maximum profit using at most k buy/sell transactions.
You must sell before buying again.
""",
        "tests": """
from solution import max_profit_k_transactions

assert max_profit_k_transactions([2, 4, 1], 2) == 2
assert max_profit_k_transactions([3, 2, 6, 5, 0, 3], 2) == 7
assert max_profit_k_transactions([], 3) == 0
assert max_profit_k_transactions([1, 2, 3, 4, 5], 100) == 4
""",
    },
    # Extra tasks added for a stronger comparison.
    {
        "id": "longest_increasing_subsequence",
        "prompt": """
Implement:

def longest_increasing_subsequence(nums: list[int]) -> int

Return the length of the longest strictly increasing subsequence.
""",
        "tests": """
from solution import longest_increasing_subsequence

assert longest_increasing_subsequence([10,9,2,5,3,7,101,18]) == 4
assert longest_increasing_subsequence([0,1,0,3,2,3]) == 4
assert longest_increasing_subsequence([7,7,7,7]) == 1
assert longest_increasing_subsequence([]) == 0
""",
    },
    {
        "id": "decode_ways",
        "prompt": """
Implement:

def decode_ways(s: str) -> int

Digits map to letters as 1 -> A, 2 -> B, ..., 26 -> Z.
Return the number of valid decodings of s.
A string starting with 0 is invalid.
""",
        "tests": """
from solution import decode_ways

assert decode_ways("12") == 2
assert decode_ways("226") == 3
assert decode_ways("06") == 0
assert decode_ways("11106") == 2
assert decode_ways("") == 0
""",
    },
    {
        "id": "subarray_sum_equals_k",
        "prompt": """
Implement:

def subarray_sum_equals_k(nums: list[int], k: int) -> int

Return the number of contiguous subarrays whose sum is exactly k.
The input may contain negative numbers.
""",
        "tests": """
from solution import subarray_sum_equals_k

assert subarray_sum_equals_k([1, 1, 1], 2) == 2
assert subarray_sum_equals_k([1, 2, 3], 3) == 2
assert subarray_sum_equals_k([1, -1, 0], 0) == 3
assert subarray_sum_equals_k([], 0) == 0
""",
    },
    {
        "id": "kth_largest",
        "prompt": """
Implement:

def kth_largest(nums: list[int], k: int) -> int

Return the kth largest element in nums.
Duplicates count as separate elements.
""",
        "tests": """
from solution import kth_largest

assert kth_largest([3,2,1,5,6,4], 2) == 5
assert kth_largest([3,2,3,1,2,4,5,5,6], 4) == 4
assert kth_largest([1], 1) == 1
assert kth_largest([-1, -2, -3], 1) == -1
""",
    },
]
