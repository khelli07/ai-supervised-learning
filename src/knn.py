import numpy as np
import pandas as pd
from collections import Counter


class KNN:
    def __init__(self, data, k, is_clf=False):
        self.data = data
        self.k = k
        self.is_clf = is_clf

    def predict(self, X):
        distance_and_indices = []

        for i, data in enumerate(self.data):
            dist = self.__euclidean_distance(data[:-1], X)
            distance_and_indices.append((dist, i))

        top_k_distance = sorted(distance_and_indices, key=lambda x: x[0])[: self.k]
        k_nearest_labels = [self.data[i][-1] for _, i in top_k_distance]

        if self.is_clf:
            return self.__most_frequent(k_nearest_labels)
        else:
            return self.__average(k_nearest_labels)

    def __average(self, labels):
        return np.average(labels)

    def __most_frequent(self, labels):
        return Counter(labels).most_common(1)[0][0]

    def __euclidean_distance(self, p1, p2):
        sum_of_squared = 0
        for i in range(len(p1)):
            sum_of_squared += np.square(p1[i] - p2[i])
        return np.sqrt(sum_of_squared)


# ===== Uncomment if you want to try to run this =====
# Data are assumed to be cleaned first
# data = pd.read_csv("./data/sample_data.csv")
# data = np.array(data)

# X = [1, 0, 1, 1]

# print(KNN(np.array(data), 5, is_clf=True).predict(X))
