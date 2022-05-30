from abc import ABC, abstractmethod
import Node


class Visitor(ABC):
    # It follows the
    # https://refactoring.guru/design-patterns/visitor
    @abstractmethod
    def visit_parent(self, parent: Node):
        pass

    @abstractmethod
    def visit_leaf(self, leaf: Node):
        pass


class FeatureImportance(Visitor):
    def __init__(self):
        self.occurrences = {}

    def visit_parent(self, parent: Node) -> None:
        k = parent.feature_index
        if k in self.occurrences.keys():
            self.occurrences[k] += 1
        else:
            self.occurrences[k] = 1

        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)

    def visit_leaf(self, leaf: Node) -> None:
        pass


class PrinterTree(Visitor):
    def __init__(self):
        self.depth = 0

    def visit_parent(self, parent: Node) -> None:
        print("\t"*self.depth+"parent, {}, {}".format(
            parent.feature_index, parent.value))
        self.depth += 1
        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)
        self.depth -= 1

    def visit_leaf(self, leaf: Node) -> None:
        print("\t"*self.depth + "leaf, {}".format(
            leaf.value_or_label_or_depth))
