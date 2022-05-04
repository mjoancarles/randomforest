from abc import ABC, abstractmethod

class Visitor(ABC):
    @abstractmethod
    def visit_parent(self,parent):
        pass

    @abstractmethod
    def visit_leaf(self,leaf):
        pass

class Feature_importance(Visitor):
    def __init__(self):
        self.occurrences={}

    def visit_parent(self,parent):
        k=parent.feature_index
        if k in self.occurrences.keys():
            self.occurrences[k]+=1
        else:
            self.occurrences[k]=1

        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)

    def visit_leaf(self,leaf):
        pass

class Printer_tree(Visitor):
    def __init__(self):
        self.depth=0

    def visit_parent(self,parent):
        print("\t"*self.depth+"parent, {}, {}".format(parent.feature_index,parent.value))
        self.depth+=1
        parent.left_child.accept_visitor(self)
        parent.right_child.accept_visitor(self)
        self.depth-=1

    def visit_leaf(self, leaf):
        print("\t"*self.depth + "leaf, {}".format(leaf.value_or_label))