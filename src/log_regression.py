import numpy as np
import pandas as pd


class LogisticRegression:  # is a Perceptron
    def __init__(self):
        self.coeff = None
        self.bias = None

    def fit(self, X, y, epochs, learning_rate=0.05):
        # initialize
        self.lr = learning_rate
        self.coeff = np.random.normal(size=(1, X.shape[1]))
        self.bias = 0

        # train
        for i in range(epochs):
            ypred = self.predict(X)

            delta = ypred - y
            self.__update_coeff(X, delta)

            loss = self.__loss(ypred.reshape(-1, 1)[0], y)
            print(f"Epoch {i + 1}/{epochs}, loss:", loss)

    def predict(self, X):
        return self.__sigmoid(self.__activate(X))

    def classify_prediction(self, prediction, threshold=0.5):
        return np.array(prediction > threshold, dtype=np.int32)

    def __update_coeff(self, X, delta):
        self.coeff = self.coeff - self.lr * (delta @ X)
        self.bias = self.bias - self.lr * (delta)
        self.bias = np.average(self.bias)

    def __activate(self, X):
        return self.coeff @ X.T + self.bias

    def __sigmoid(self, inputs):
        logmax = 709.78
        inputs[-inputs > logmax] = logmax
        return 1.0 / (1.0 + np.exp(-inputs, dtype=np.float64))

    def __loss(self, ypred, ytrue):
        return np.sum(
            np.nan_to_num(-ytrue * np.log(ypred) - (1 - ytrue) * np.log(1 - ypred))
        )

# ===== Uncomment if you want to try to run this =====
# Data are assumed to be cleaned first
# data = pd.read_csv("./data/sample_data.csv")
# data = np.array(data)
# X = data[:, :-1]
# y = data[:, -1]

# np.random.seed(300)
# model = LogisticRegression()
# model.fit(X, y, epochs=5000, learning_rate=1e-4)
# print(model.classify_prediction(model.predict(X)))
