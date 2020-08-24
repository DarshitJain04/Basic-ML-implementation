import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Creating dataset
X, Y = datasets.make_regression(n_samples=200, n_features=1, noise=30, random_state=123)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)



class LinearRegression():

    def __init__(self, learning_rate = 0.001, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X_train, Y_train):
        samples, features = X_train.shape
        self.weights = np.zeros(features)
        self.bias = 0

        print('Dimension of X :', X_train.shape)
        print('Dimension of Y :', Y_train.shape)
        print('Dimension of Weights :', self.weights.shape)

        # weights.shape = (1, )
        # X_train.shape = (160, 1)
        # Y_train.shape = (160, )

        for i in range(self.epochs):

            y = np.dot(X_train, self.weights) + self.bias

            # Here X_train.T is taken because when multiplying two matrices, say A(lxm) and B(mxn), 
            # columns in A should be same as rows in B
            # So X_train.T.shape = (1, 160) and (y- Y_train).shape = (160, )
            
            dw = (2/samples) * np.dot(X_train.T, (y - Y_train))
            db = (2/samples) * np.sum((y - Y_train))

            self.weights = self.weights - self.learning_rate*dw
            self.bias = self.bias - self.learning_rate*db
    
    def predict(self, X_test):
        y = np.dot(X_test, self.weights) + self.bias
        return y


clf = LinearRegression()
clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)


print('Mean squared error :', np.mean((Y_test - prediction)**2))