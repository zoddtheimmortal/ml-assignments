import math
import pandas as pd
from collections import deque

class Node:
    def __init__(self, feature=None, threshold=None, value=None, children=None):
        self.feature = feature # feature on which its split
        self.threshold = threshold # numerical val at which its split
        self.value = value # label of leaf nodes
        self.children = children if children is not None else {}

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None

    def fit(self, X, y):
        data = X.copy()

        # label = predicted column
        data['label'] = y
        self.root = self._build_tree(data)

    def _build_tree(self, data):

        # all data points have same label,make node that value
        if len(data['label'].unique()) == 1:
            return Node(value=data['label'].iloc[0])

        # no more features to split on, make node with most common label
        if data.shape[1] == 1:
            return Node(value=data['label'].mode()[0])

        best_feature, best_threshold = self._find_best_split(data)

        # no more information gain, make node with most common label
        if best_feature is None:
            return Node(value=data['label'].mode()[0])

        node = Node(feature=best_feature, threshold=best_threshold)

        # split data acc to <= and > threshold
        left_data = data[data[best_feature] <= best_threshold].drop(columns=[best_feature])
        right_data = data[data[best_feature] > best_threshold].drop(columns=[best_feature])
        
        node.children['left'] = self._build_tree(left_data)
        node.children['right'] = self._build_tree(right_data)

        return node

    def _find_best_split(self, data):
        best_feature = None
        best_threshold = None
        best_info_gain = -float('inf')

        for feature in data.columns[:-1]:
            thresholds = data[feature].unique()

            # check info gain for each unique value of feature
            for threshold in thresholds:
                info_gain = self._information_gain(data, feature, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _entropy(self, data):
        labels = data['label']
        label_counts = labels.value_counts()
        entropy = 0
        for count in label_counts:
            probability = count / len(labels)
            entropy -= probability * math.log2(probability)
        return entropy

    def _information_gain(self, data, feature, threshold):
        parent_entropy = self._entropy(data)
        left_data = data[data[feature] <= threshold]
        right_data = data[data[feature] > threshold]

        if len(left_data) == 0 or len(right_data) == 0:
            return 0

        # info gain = parent entropy - weighted avg of child entropies
        left_entropy = self._entropy(left_data)
        right_entropy = self._entropy(right_data)
        weighted_avg_entropy = (len(left_data) / len(data)) * left_entropy + (len(right_data) / len(data)) * right_entropy

        return parent_entropy - weighted_avg_entropy

    def predict(self, X):
        return X.apply(self._predict_row, axis=1)

    def _predict_row(self, row):
        node = self.root
        
        # search for label like binary tree search
        while node.value is None:
            if row[node.feature] <= node.threshold:
                node = node.children['left']
            else:
                node = node.children['right']
        return node.value

    def export_tree(self):
        return self._recurse(self.root)
    
    def _recurse(self,node):
        if node.value is not None:
            return node.value
        return {
            'feature': node.feature,
            'threshold': node.threshold,
            'left': self._recurse(node.children['left']),
            'right': self._recurse(node.children['right'])
        }