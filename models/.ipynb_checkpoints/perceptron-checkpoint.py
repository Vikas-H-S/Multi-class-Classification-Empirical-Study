"""Perceptron model."""

import numpy as np
from operator import add, sub

class Perceptron:
    def __init__(self, n_class: int, lr: float, decay:float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.initial_lr = lr
        self.decay = decay
        self.epochs = epochs
        self.n_class = n_class
        self.mean = None
        self.std = None

    def update_learning_rate(self, epoch: int):
        """Apply learning rate decay based on the epoch."""
        self.lr = self.lr / (1 + self.decay * epoch)

    def data_normalization(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        if self.std is None:
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        X_train = self.data_normalization(X_train) / 255.0
        
        N, D = X_train.shape
        self.w = np.zeros((self.n_class, D + 1))  # Initialize weight matrix
        
        for epoch in range(self.epochs):
            self.update_learning_rate(epoch)

            indices = np.arange(N)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            for x, y_true in zip(X_train, y_train):
                x = np.append(x, 1)
                scores = np.dot(self.w, x)
                y_pred = np.argmax(scores)

                if y_pred != y_true:
                    self.w[y_true] += self.lr * x
                    self.w[y_pred] -= self.lr * x

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = self.data_normalization(X_test) / 255.0
        ones = np.ones((X_test.shape[0], 1))
        X_test = np.hstack((X_test, ones))
        scores = np.dot(X_test, self.w.T)
        predictions = np.argmax(scores, axis=1)
        return predictions
