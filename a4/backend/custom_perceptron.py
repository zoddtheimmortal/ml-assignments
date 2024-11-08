import numpy as np


class CustomPerceptron:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def activation_fn(self, x: float) -> int:
        return 1 if x >= 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        N, D = X.shape

        np.random.seed(68)
        self.weights = np.random.randn(D)
        self.bias = 0.01

        for _ in range(self.epochs):
            for i in range(N):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation_fn(linear_output)

                update = self.learning_rate * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = np.array([self.activation_fn(x) for x in linear_output])
        return y_pred