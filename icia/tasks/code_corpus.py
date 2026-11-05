"""
Phase 16 corpus: correct reference implementations used to train the real
autoregressive code decoder that serves as the governed-decoding baseline.
Each challenge solution satisfies its own prohibitions and passes its tests.
"""

from __future__ import annotations

CHALLENGE_SOLUTIONS: dict[str, str] = {
    "MBPP-C01": '''def fibonacci(n):
    """Return the n-th Fibonacci number iteratively."""
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
''',
    "MBPP-C02": '''def reverse_inplace(arr):
    """Reverse the list in place using two pointers."""
    i, j = 0, len(arr) - 1
    while i < j:
        arr[i], arr[j] = arr[j], arr[i]
        i += 1
        j -= 1
    return arr
''',
    "MBPP-C03": '''def two_sum(nums, target):
    """Return indices of the pair summing to target in one pass."""
    seen = {}
    for idx, val in enumerate(nums):
        other = target - val
        if other in seen:
            return [seen[other], idx]
        seen[val] = idx
    return []
''',
    "MBPP-C04": '''def top_k_words(words, k):
    """Return the k most frequent words using a plain dict."""
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:k]]
''',
    "MBPP-C05": '''def check_brackets(code):
    """Return True if all parentheses and brackets match."""
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in code:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != pairs[ch]:
                return False
    return len(stack) == 0
''',
}

AUX_FUNCTIONS: list[tuple[str, str]] = [
    (
        "Write a function `is_palindrome(s)` that returns True if the string equals its reversal.",
        '''def is_palindrome(s):
    """Return True when s reads the same backwards."""
    n = len(s)
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True
''',
    ),
    (
        "Write a function `factorial(n)` returning n! as an integer.",
        '''def factorial(n):
    """Compute n factorial iteratively."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
''',
    ),
    (
        "Write a function `running_mean(xs)` returning prefix means of a list of floats.",
        '''def running_mean(xs):
    """Return prefix averages of the input sequence."""
    out = []
    total = 0.0
    for i, x in enumerate(xs):
        total += x
        out.append(total / (i + 1))
    return out
''',
    ),
    (
        "Write a function `caesar(text, shift)` applying a Caesar cipher to lowercase letters.",
        '''def caesar(text, shift):
    """Shift each lowercase letter by the given amount."""
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)
''',
    ),
    (
        "Write a function `flatten(matrix)` turning a list of lists into one list.",
        '''def flatten(matrix):
    """Flatten a matrix into a single list."""
    out = []
    for row in matrix:
        for item in row:
            out.append(item)
    return out
''',
    ),
    (
        "Write a function `count_vowels(s)` counting vowels in a string.",
        '''def count_vowels(s):
    """Count the vowels in the given text."""
    total = 0
    for ch in s.lower():
        if ch in "aeiou":
            total += 1
    return total
''',
    ),
    (
        "Write a function `dedupe(xs)` removing duplicates while preserving order.",
        '''def dedupe(xs):
    """Return the list without duplicates, keeping first occurrences."""
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
''',
    ),
    (
        "Write a function `dot_product(a, b)` computing the sum of elementwise products.",
        '''def dot_product(a, b):
    """Compute the dot product of two equal-length lists."""
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
''',
    ),
    (
        "Write a function `word_count(sentence)` counting whitespace-separated tokens.",
        '''def word_count(sentence):
    """Count the number of words in the sentence."""
    words = []
    current = ""
    for ch in sentence + " ":
        if ch == " ":
            if current:
                words.append(current)
                current = ""
        else:
            current += ch
    return len(words)
''',
    ),
    (
        "Write a function `clip(value, lo, hi)` clamping a number into a range.",
        '''def clip(value, lo, hi):
    """Clamp value between lo and hi."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
''',
    ),
    (
        "Write a function `transpose(matrix)` returning the transposed matrix.",
        '''def transpose(matrix):
    """Return the transpose of a rectangular matrix."""
    if not matrix:
        return []
    rows = len(matrix)
    cols = len(matrix[0])
    out = []
    for c in range(cols):
        out.append([matrix[r][c] for r in range(rows)])
    return out
''',
    ),
    (
        "Write a function `binary_search(sorted_nums, target)` returning the index or -1.",
        '''def binary_search(sorted_nums, target):
    """Locate target in a sorted list using binary search."""
    lo = 0
    hi = len(sorted_nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if sorted_nums[mid] == target:
            return mid
        elif sorted_nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
    ),
]


def build_training_pairs(challenge_prompts: dict[str, str]) -> list[tuple[str, str]]:
    """Pairs of (prompt, code) across challenges plus auxiliary generalization functions."""
    pairs = [(prompt, CHALLENGE_SOLUTIONS[task_id]) for task_id, prompt in challenge_prompts.items()]
    pairs.extend(AUX_FUNCTIONS)
    return pairs
