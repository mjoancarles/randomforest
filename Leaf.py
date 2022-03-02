from Node import Node

class Leaf(Node):
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label
