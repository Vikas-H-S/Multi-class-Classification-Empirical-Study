"""Softmax model."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from operator import add, sub

class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.mean = None
        self.std = None
        self.scaler = StandardScaler()

    def data_normalization(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        if self.std is None:
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        N = X_train.shape[0]
        logits = np.dot(X_train, self.w)
        probs = self.softmax(logits)
        
        # One-hot encoding of y_train
        y_one_hot = np.zeros_like(probs)
        y_one_hot[np.arange(N), y_train] = 1
        
        # Gradient of the loss with respect to weights
        grad = np.dot(X_train.T, probs - y_one_hot) / N
        
        # Add regularization term
        grad += self.reg_const * self.w
        
        return grad

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        # X_train = self.scaler.fit_transform(X_train)
        X_train = self.data_normalization(X_train)
        
        N, D = X_train.shape
        self.w = np.random.randn(D, self.n_class) * 0.01  # Initialize weights

        for epoch in range(self.epochs):
            grad = self.calc_gradient(X_train, y_train)
            self.w -= self.lr * grad  # Update weights

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
        # X_test = self.scaler.transform(X_test)
        X_test = self.data_normalization(X_test)
        
        logits = np.dot(X_test, self.w)
        probs = self.softmax(logits)
        return np.argmax(probs, axis=1)
