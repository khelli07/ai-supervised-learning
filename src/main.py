import pandas as pd
import numpy as np
import os

from id_3 import ID3
from knn import KNN
from log_regression import LogisticRegression

files = os.listdir("./data")
for i in range(len(files)):
    print(f"{i + 1}. {files[i]}")

file_index = int(input("What data do you want to train? "))
if file_index < 1 or file_index > len(files):
    raise ValueError("Invalid choice of data")

filename = files[file_index - 1]
data = pd.read_csv(f"./data/{filename}")
data = np.array(data)

algorithms = ["Decision Tree with ID 3", "K-Nearest Neighbours", "Logistic Regression"]
for i in range(3):
    print(f"{i + 1}. {algorithms[i]}")

algo_index = int(input("What algorithm you want to use? "))
if algo_index < 1 or algo_index > 3:
    raise ValueError("Invalid choice of algorithm")

if algo_index == 1:
    model = ID3()
    model.fit(np.array(data))
    print("Here's your tree: ")
    model.print_tree(model.root, 0)

elif algo_index == 2:
    k = int(input("Please input k-value: "))
    is_clf = input("Is it classification? (Y/N) ")
    is_clf = True if is_clf == "y" or is_clf == "Y" else False
    model = KNN(data, k, is_clf)

    to_predict = input("Please input 1 data to be predicted (comma separated): ")
    X = to_predict.split(",")
    if len(X) > data.shape[1] - 1:
        raise ValueError("That data is too much!")
    X = np.array(X, dtype=np.float64)
    print("Predicted =", model.predict(X))

elif algo_index == 3:
    epochs = int(input("Epochs = "))
    lr = float(input("Learning rate = "))

    X = data[:, :-1]
    y = data[:, -1]
    model = LogisticRegression()
    model.fit(X, y, epochs=epochs, learning_rate=lr)
