import numpy as np
import pandas as pd
import random
from collections import Counter


class ID3Leaf:
    def __init__(self, verdict):
        self.verdict = verdict


class ID3Node:
    def __init__(self, data, col_idx, true, false):
        self.data = data
        self.col_idx = col_idx
        self.true = true
        self.false = false


class ID3:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self, data):
        column_left = [i for i in range(len(data[0]) - 1)]
        self.root = self.__grow_tree(data, column_left, 0)

    def predict(self, X):
        return self.__classify(self.root, X)

    def print_tree(self, node, indent):
        spacing = " " * indent
        if isinstance(node, ID3Leaf):
            print(spacing + "PREDICTED: " + str(node.verdict))
        else:
            print(spacing + "COLUMN = " + str(node.col_idx))
            print(spacing + "|=== TRUE ===|")
            self.print_tree(node.true, indent + 4)

            print(" " * indent + "|=== FALSE ===|")
            self.print_tree(node.false, indent + 4)

    def __classify(self, node, data):
        if isinstance(node, ID3Leaf):
            return node.verdict
        else:
            if data[node.col_idx] == 1:
                return self.__classify(node.true, data)
            else:
                return self.__classify(node.false, data)

    def __partition(self, data, col_idx):
        true = data[np.where(data[:, col_idx] == 0)]
        false = data[np.where(data[:, col_idx] == 1)]
        return true, false

    def __get_most_common(self, data):
        counts = Counter(data)
        common = counts.most_common(1)
        try:
            return common[0][0]
        except Exception:
            return random.randint(0, 1)

    def __grow_tree(self, data, column_left, current_depth):
        if not column_left or (self.max_depth and current_depth >= self.max_depth):
            most_common = self.__get_most_common(data[:, -1])
            return ID3Leaf(most_common)

        best_ig_col = self.__choose_best_ig(data, column_left)
        true, false = self.__partition(data, best_ig_col)

        true_branch = self.__grow_tree(
            true, [col for col in column_left if col != best_ig_col], current_depth + 1
        )
        false_branch = self.__grow_tree(
            false, [col for col in column_left if col != best_ig_col], current_depth + 1
        )

        return ID3Node(data, best_ig_col, true_branch, false_branch)

    def __choose_best_ig(self, data, column_left):
        pool = []

        for col in column_left:
            true, false = self.__partition(data, col)
            ig = self.__information_gain(data, true, false)
            pool.append((ig, col))

        best_ig = max(pool, key=lambda x: x[0])
        return best_ig[1]

    def __information_gain(self, total, true, false):
        total, total_entropy = self.__entropy(total[:, -1])
        t_total, true_entropy = self.__entropy(true[:, -1])
        f_total, false_entropy = self.__entropy(false[:, -1])

        return (
            total_entropy
            - (t_total / total * true_entropy)
            - (f_total / total * false_entropy)
        )

    def __entropy(self, data):
        counts = Counter(data)
        total = sum(counts.values())
        proba = {k: v / total for k, v in counts.items()}

        return total, np.sum([-p * np.log2(p) for p in proba.values()])


# ===== Uncomment if you want to try to run this =====
# Data are assumed to be cleaned first
# data = pd.read_csv("./data/sample_data.csv")
# data = np.array(data)

# tree = ID3()
# tree.fit(np.array(data))
# tree.print_tree(tree.root, 0)
# print(tree.predict(np.array([0, 1, 0])))
