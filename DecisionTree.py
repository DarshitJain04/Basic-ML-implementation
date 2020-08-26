import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load data 
data = datasets.load_breast_cancer()
# Here the dataset contains only 2 classes
X, Y = data.data, data.target

# X.shape = (569, 30) Y.shape = (569,)
# (569, 30) => 569 samples with 30 features

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1234)

# Entropy
def entropy(y):
    occurance = np.bincount(y)
    P = occurance / len(y)
    entropy = -np.sum([p * np.log2(p) for p in P if p > 0])
    return entropy

# If the data is homogenous, entropy = 0 and when dataset have is equally divided in two different parts, entropy = 1

# The information gain is based on the decrease in entropy after a dataset is split according to an attribute.


class Node():

    def __init__(
        self,
        split_feature = None,
        split_threshold = None,
        left = None,
        right = None,
        *,
        value = None
    ):

        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left = left
        self.right = right
        self.value = value

    def leaf_node(self):
        return self.value is not None

class DecisionTree():

    def __init__(self, min_sample = 2, max_depth = 100, n_features = None):

        self.min_sample = min_sample
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
    
    def fit(self, X_train, Y_train):
        self.n_features = X_train.shape[1] if not self.n_features else min(self.n_features, X_train.shape[1])
        self.root = self.grow_tree(X_train, Y_train)
    
    def grow_tree(self, X, Y, depth = 0):
        samples, features = X.shape
        n_labels = len(np.unique(Y))

        # Check if the node is leaf or not
        if (depth >= self.max_depth or n_labels == 1 or samples < self.min_sample):
            leaf_label = self.most_common_label(Y)
            return Node(value = leaf_label)
        
        # Selecting column index at random 
        feature_indexes = np.random.choice(features, self.n_features, replace = False)

        # Greedy Search
        best_feature, best_threshold = self.best_criteria(X, Y, feature_indexes)

        left_indexes, right_indexes = self.split(X[:, best_feature], best_threshold)

        left = self.grow_tree(X[left_indexes, :], Y[left_indexes], depth+1)
        right = self.grow_tree(X[right_indexes, :], Y[right_indexes], depth+1)

        return Node(best_feature, best_threshold, left, right)
    
    def best_criteria(self, X, Y, feature_indexes):
        best_info_gain = -1
        split_index, split_threshold = None, None
        for feature_index in feature_indexes:
            X_column  = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(X_column, Y, threshold)

                if gain > best_info_gain :
                    best_info_gain = gain
                    split_index = feature_index
                    split_threshold = threshold
        
        return split_index, split_threshold
        # Returns the best split feature and the threshold value which gives us maximum info
    
    def information_gain(self, X_column, Y, split_threshold):
        # Parent entropy
        parent_entropy = entropy(Y)
        # Generate split to find out children node's entropy
        left_indexes, right_indexes = self.split(X_column, split_threshold)

        if len(left_indexes) == 0 or len(right_indexes) == 0 :
            return 0
        
        len_left_samples, len_right_samples = len(left_indexes), len(right_indexes)
        entropy_left, entropy_right = entropy(Y[left_indexes]), entropy(Y[right_indexes])

        child_entropy = (len_left_samples/len(Y)) * entropy_left + (len_right_samples/len(Y)) * entropy_right

        information_gain = parent_entropy - child_entropy

        return information_gain
    
    def split(self, X_column, split_threshold):
        left_indexes = np.argwhere(X_column <= split_threshold).flatten()
        right_indexes = np.argwhere(X_column > split_threshold).flatten()
        return left_indexes, right_indexes
    
    def most_common_label(self, Y):
        most_common = Counter(Y).most_common(1)[0][0]
        return most_common

    def predict(self, X_test):
        return np.array([self.traverse_tree(x_test, self.root) for x_test in X_test])
    
    def traverse_tree(self, x_test, node):
        if node.leaf_node():
            return node.value
        if x_test[node.split_feature] <= node.split_threshold:
            return self.traverse_tree(x_test, node.left)
        return self.traverse_tree(x_test, node.right)


clf = DecisionTree(max_depth=10)
clf.fit(X_train, Y_train)

prediction = clf.predict(X_test)

accuracy = np.sum(Y_test == prediction) / len(Y_test)

print('Accuracy :', accuracy*100)
