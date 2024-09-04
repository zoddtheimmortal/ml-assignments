import math
from collections import deque

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.children = None

class DecisionTreeClassifier:
    def __init__(self):
        self.root = None
