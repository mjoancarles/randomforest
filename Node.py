from abc import ABC, abstractmethod

class Node(ABC):

    @abstractmethod
    def predict(self,x):
        pass
