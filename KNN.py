import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from collections import Counter


# KNN is a simple machine learning algorithm used for classification tasks


# Loading the dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target


# Splitting the dataset into training and test data
# test_size = 0.2 represents the proportion of the dataset to include in the test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)


# Calculating the Euclidian Distance between the test point and the training data
def euclidian_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))


# Model
class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predicted_labels = [self.predictor(x) for x in X_test]
        return np.array(predicted_labels)
        
    def predictor(self, x):
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]
        # Getting the indices of k nearest points
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]
        most_common_label = Counter(k_nearest_labels).most_common(1)
        return most_common_label[0][0]


# We can check for different values of k
clf = KNN(k=3)
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)


# Accuracy changes based on value of k and the amount of test data
accuracy = np.sum( predictions == y_test ) / len(y_test)
print('Accuracy :', accuracy*100,'%')

