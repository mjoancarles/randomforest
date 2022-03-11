from ImpurityMeasures import ImpurityMeasures
import numpy as np


class Gini(ImpurityMeasures):
    def compute_impurity(self, dataset):
        count = np.bincount(dataset.y)
        gini = 1-sum(list(map(lambda x: (x/dataset.num_samples)**2, count)))
        return gini