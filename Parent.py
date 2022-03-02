from Node import Node


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


