from numpy import ndarray
from abc import ABC, abstractmethod
import Visitor


class Node(ABC):
    # It follows the composite design pattern
    # https://refactoring.guru/design-patterns/composite
    @abstractmethod
    def predict(self, x: ndarray):
        pass

    @abstractmethod
    def accept_visitor(self, visitor: Visitor):
        pass


class Leaf(Node):
    def __init__(self, value_or_label_or_depth: float):
        self.value_or_label_or_depth = value_or_label_or_depth

    def predict(self, x: ndarray) -> float:
        return self.value_or_label_or_depth

    def accept_visitor(self, visitor: Visitor) -> None:
        visitor.visit_leaf(self)


class Parent(Node):
    def __init__(self, feature_index: int, value: float):
        self.feature_index = feature_index
        self.value = value
        self.left_child = None
        self.right_child = None

    def predict(self, x: ndarray) -> Node:
        if x[self.feature_index] <= self.value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def accept_visitor(self, visitor: Visitor) -> None:
        visitor.visit_parent(self)
