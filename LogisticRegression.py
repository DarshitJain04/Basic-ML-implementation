import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Loading the dataset
iris = datasets.load_iris()

# We're considering only one feature and using the classifer as a binary classifer 
# To predict if the flower is Iris-Virginica or not

# Slicing data and considering only one out of four features
X = iris.data[:, 3:]
print('X.shape =', X.shape)

# Changing lables to 0 if not Iris-Virginica and to 1 if Iris-Virginica
Y = (iris.target == 2).astype(np.int)
print('Y.shape =', Y.shape)


# Splitting the dataset into training and test data
# test_size = 0.2 represents the proportion of the dataset to include in the test dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 123)

class LogisticRegression():

    def __init__(self, learning_rate = 0.001, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, Y_train):
        samples, features = X_train.shape
        self.weights = np.zeros(features)
        print('weights.shape =', self.weights.shape)
        self.bias = 0

        for i in range(self.epochs):

            reg = np.dot(X_train, self.weights) + self.bias
            y_predicted = self.sigmoid(reg)
            dw = (2/samples) * np.dot(X_train.T, (y_predicted - Y_train))
            db = (2/samples) * np.sum((y_predicted - Y_train))

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
    
    def predict(self, X_test):
        reg = np.dot(X_test, self.weights) + self.bias
        y = self.sigmoid(reg)
        predict_class = [1 if i>0.5 else 0 for i in y]
        return predict_class
    
    # Sigmoid function returns a value between 0 and 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

# Increasing the number of epochs results in better approximations, hence results in a better curve and accuracy
clf = LogisticRegression(epochs=5000)
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)

accuracy = np.sum( prediction == Y_test ) / len(Y_test)
print('Accuracy :', accuracy*100)