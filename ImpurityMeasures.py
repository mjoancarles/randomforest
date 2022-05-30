import numpy as np
from abc import ABC, abstractmethod
from Dataset import Dataset


class ImpurityMeasures(ABC):
    # It follows the strategy design pattern
    # https://refactoring.guru/design-patterns/strategy
    @abstractmethod
    def compute_impurity(self, dataset: Dataset) -> float:
        pass


class Gini(ImpurityMeasures):
    def compute_impurity(self, dataset: Dataset) -> float:
        prob = dataset.probability()
        gini = 1 - np.sum(prob ** 2)
        return gini


class Entropy(ImpurityMeasures):
    def compute_impurity(self, dataset: Dataset) -> float:
        prob = dataset.probability()
        filtered_prob = np.array([x for x in prob if x != 0])
        return -np.sum(filtered_prob*np.log(filtered_prob))


class SumSquareError(ImpurityMeasures):
    def compute_impurity(self, dataset: Dataset) -> float:
        y_left = np.mean(dataset.y)
        return +np.sum((dataset.y-y_left)**2)
