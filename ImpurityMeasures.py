from abc import ABC,abstractmethod
import numpy as np

class ImpurityMeasures(ABC):
    @abstractmethod
    def compute_impurity(self,dataset):
        pass

class Gini(ImpurityMeasures):
    def compute_impurity(self, dataset):
        prob = dataset.probability()
        gini = 1 - np.sum(prob ** 2)
        return gini

class Entropy(ImpurityMeasures):
    def compute_impurity(self,dataset):
        prob=dataset.probability()
        filtered_prob=np.array([x for x in prob if x!=0])
        return - np.sum(filtered_prob*np.log(filtered_prob))

class SumSquareError(ImpurityMeasures):
    def compute_impurity(self,dataset):
        y_left=np.mean(dataset.y)
        return np.sum((dataset.y-y_left)**2)