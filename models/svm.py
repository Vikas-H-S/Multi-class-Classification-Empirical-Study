"""Support Vector Machine (SVM) model."""

import numpy as np
# from sklearn.preprocessing import StandardScaler
from operator import add, sub

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.alpha = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.mean = None
        self.std = None
        # self.scaler = StandardScaler()

    def data_normalization(self, X):
        if self.mean is None:
            self.mean = np.mean(X, axis=0)
        if self.std is None:
            self.std = np.std(X, axis=0)
        return (X - self.mean) / (self.std + 1e-8)

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N, D = X_train.shape
        # Initialize the gradient as zero
        grad_w = np.zeros(self.w.shape)

        for i in range(N):
            scores = X_train[i].dot(self.w)
            correct_class_score = scores[y_train[i]]
            for j in range(self.n_class):
                if j == y_train[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # Hinge loss
                if margin > 0:
                    grad_w[:, j] += X_train[i]
                    grad_w[:, y_train[i]] -= X_train[i]

        # Apply regularization
        grad_w /= N
        grad_w += self.reg_const * self.w
        return grad_w

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
        self.w = np.random.randn(D, self.n_class) * 0.001  # Initialize weights
        
        # Iterate over the number of epochs
        for epoch in range(self.epochs):
           
            # Compute the gradient
            grad_w = self.calc_gradient(X_train, y_train)

            # Update the weights
            self.w -= self.alpha * grad_w

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
        scores = X_test.dot(self.w)
        return np.argmax(scores, axis=1)
