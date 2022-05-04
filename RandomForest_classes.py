from RandomForest import RandomForest
import logging.config
from ImpurityMeasures import *
from Node import Leaf

logging.config.fileConfig("logging.conf")
logger=logging.getLogger("RandomForest")

class RandomForestClassifier(RandomForest):
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, name_impurity, multiprocessing, extra_trees):
        super().__init__(num_trees, min_size, max_depth, ratio_samples, num_random_features, name_impurity, multiprocessing, extra_trees)

    def _make_impurity(self, name_impurity):
        if name_impurity.lower() == "gini":
            return Gini()
        elif name_impurity.lower() == "entropy":
            return Entropy()

    def _combine_predictions(self, predictions):
        return np.argmax(np.bincount(predictions))

    def _make_leaf(self, dataset): # most frequent class in dataset
        return Leaf(dataset.most_frequent_label())

class RandomForestRegressor(RandomForest):
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, name_impurity, multiprocessing, extra_trees):
        super().__init__(num_trees, min_size, max_depth, ratio_samples, num_random_features, name_impurity, multiprocessing, extra_trees)

    def _make_impurity(self, name_impurity):
        if name_impurity.lower() == "sum square error":
            return SumSquareError()

    def _combine_predictions(self, predictions):
        return np.mean(predictions)

    def _make_leaf(self,dataset):
        return Leaf(dataset.mean_value())