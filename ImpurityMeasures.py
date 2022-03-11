from abc import ABC,abstractmethod

class ImpurityMeasures(ABC):
    @abstractmethod
    def compute_impurity(self,dataset):
        pass
