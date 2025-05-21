import os
import numpy as np


def longest_consecutive_match(vec1, vec2):
    max_len = 0
    len1, len2 = len(vec1), len(vec2)

    # Use a dynamic programming (DP) table where dp[i][j] means
    # length of longest suffix match ending at vec1[i-1] and vec2[j-1]
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if vec1[i - 1] == vec2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
            else:
                dp[i][j] = 0

    return max_len


def longest_consecutive_match_with_tolerance(vec1, vec2, max_tolerance=1):
    max_len = 0
    max_tolerated = 0
    len1, len2 = len(vec1), len(vec2)

    for start1 in range(len1):
        for start2 in range(len2):
            length = 0
            tolerated = 0
            for offset in range(min(len1 - start1, len2 - start2)):
                if vec1[start1 + offset] != vec2[start2 + offset]:
                    tolerated += 1
                    if tolerated > max_tolerance:
                        break
                length += 1

            # Track the best length and tolerance used
            if length > max_len or (length == max_len and tolerated < max_tolerated):
                max_len = length
                max_tolerated = tolerated

    return max_len, max_tolerated


def save_in_hits(input_dir, vector):
    directory = os.fsencode(input_dir)
    best_file = None
    best_hit_count = 0
    best_hit_dict = {}

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not filename.endswith(".npz"):
            continue

        filepath = os.path.join(input_dir, filename)
        data = np.load(filepath, allow_pickle=True)
        hit_dict = {}
        total_hits = 0

        for key in data:
            vec = data[key]
            match_len = longest_consecutive_match(vector, vec)
            if match_len > 0:
                hit_dict[key] = match_len
                total_hits += 1

        if total_hits > best_hit_count:
            best_file = filename
            best_hit_count = total_hits
            best_hit_dict = hit_dict

    return best_file, best_hit_dict