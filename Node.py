from abc import ABC, abstractmethod

class Node(ABC):
    @abstractmethod
    def predict(self,x):
        pass

    @abstractmethod
    def accept_visitor(self,visitor):
        pass

class Leaf(Node):
    def __init__(self, value_or_label):
        self.value_or_label = value_or_label

    def predict(self, x):
        return self.value_or_label

    def accept_visitor(self,visitor):
        visitor.visit_leaf(self)

class Parent(Node):
    def __init__(self, feature_index,value):
        self.feature_index=feature_index
        self.value=value
        self.left_child=None
        self.right_child=None

    def predict(self, x):
        if x[self.feature_index] <= self.value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def accept_visitor(self,visitor):
        visitor.visit_parent(self)