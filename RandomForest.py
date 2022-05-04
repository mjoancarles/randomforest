from Visitor import *
import numpy as np
import time
from multiprocessing import Pool
from Dataset import Dataset
from Node import Parent
import logging.config
from tqdm import tqdm
from abc import ABC, abstractmethod

logging.config.fileConfig("logging.conf")
logger=logging.getLogger("RandomForest")

class RandomForest(ABC):
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, name_impurity, multiprocessing, extra_trees):
        self.num_trees=num_trees
        self.min_size=min_size
        self.max_depth=max_depth
        self.ratio_samples=ratio_samples
        self.num_random_features=num_random_features
        self.multiprocessing=multiprocessing
        self.extra_trees= extra_trees
        self.impurity_measure = self._make_impurity(name_impurity)

    def feature_importance(self):
        feat_imp_visitor=Feature_importance()
        for tree in self.decision_trees:
            tree.accept_visitor(feat_imp_visitor)
        return feat_imp_visitor.occurrences

    def print_trees(self):
        tree_printer = Printer_tree()
        for tree in self.decision_trees:
            tree.accept_visitor(tree_printer)

    @abstractmethod
    def _make_impurity(self, name_impurity):
        pass

    @abstractmethod
    def _combine_predictions(self, predictions):
        pass

    def predict(self,X):
        if (self.multiprocessing):
            ypred= self._predict_multiprocessing(X)
        else:
            ypred= self._predict_standard(X)
        return ypred

    def _predict_standard(self,X):
        ypred = []
        t1 = time.time()
        for x in tqdm(X, total=len(X), desc="Predict progress: "):
            predictions = [root.predict(x) for root in self.decision_trees]
            # majority voting
            ypred.append(self._combine_predictions(predictions))
        t2 = time.time()
        logger.info('Predict lasted {} seconds'.format(t2 - t1))
        return np.array(ypred)

    def _target_predict(self,x):
        predictions=[root.predict(x) for root in self.decision_trees]
        prediction=self._combine_predictions(predictions)
        logger.debug("Prediction done, result: {}".format(prediction))
        return prediction

    def _predict_multiprocessing(self,X):
        t1 = time.time()
        with Pool() as pool:
            ypred = pool.map(self._target_predict,tqdm(X,total=len(X),desc="Multiprocessing predict progress: "))
        t2 = time.time()
        logger.info('Multiprocessing predict lasted {} seconds'.format(t2 - t1))
        return np.array(ypred)

    def fit(self, X, y): #train
        #its own responsibilities
        dataset = Dataset(X, y)
        if (self.multiprocessing):
            self._make_decision_trees_multiprocessing(dataset)
        else:
            self._make_decision_trees_standard(dataset)

    def _make_decision_trees_standard(self, dataset):
        self.decision_trees = []
        t1 = time.time()
        for i in tqdm(range(self.num_trees),total=self.num_trees,desc="Fit progress: "):
            # sample a subset of the dataset with replacement using np.random.choice()
            # to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset,1) # the root of the decision tree
            self.decision_trees.append(tree)
        t2 = time.time()
        logger.info('Fit lasted {} seconds'.format(t2 - t1))


    def _target_fit(self, dataset, nproc):
        logger.debug('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self.ratio_samples)
        tree = self._make_node(subset,1)
        logger.debug('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self,dataset):
        t1=time.time()
        with Pool() as pool:
            iterable=[(dataset,nprocess) for nprocess in range(self.num_trees)]
            self.decision_trees = pool.starmap(self._target_fit,tqdm(iterable,total=len(iterable),desc="Multiprocessing fit progress: "))
        t2=time.time()
        logger.info('Multiprocessing fit lasted {} seconds'.format(t2-t1))

    def _make_node(self, dataset, depth):
        if depth == self.max_depth or dataset.num_samples <= self.min_size or len(np.unique(dataset.y)) == 1:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node

    @abstractmethod
    def _make_leaf(self, dataset):
        pass

    def _make_parent_or_leaf(self, dataset, depth):
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features), self.num_random_features, replace=False)
        best_feature_index, best_value, minimum_cost, best_split = self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is an special case : dataset has samples of at least two
            # classes but the best split is moving all samples to the left or right
            # dataset and none to the other, so we make a leaf instead of a parent
            return self._make_leaf(dataset)
        else:
            node = Parent(best_feature_index, best_value)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, idx_features, dataset):
        # find the best pair (feature, value) by exploring all possible pairs
        best_feature_index, best_value, minimum_cost, best_split = np.Inf, np.Inf, np.Inf, None
        if (self.extra_trees):
            for idx in idx_features:
                max = np.amax(dataset.X[:,idx])
                min = np.amin(dataset.X[:,idx])
                val=np.random.uniform(min,max)
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_value, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
        else:
            for idx in idx_features:
                values = np.unique(dataset.X[:,idx])
                for val in values:
                    left_dataset, right_dataset = dataset.split(idx, val)
                    cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                    if cost < minimum_cost:
                        best_feature_index, best_value, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
        return best_feature_index, best_value, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset):
        number_samples_left = left_dataset.num_samples
        number_samples_right = right_dataset.num_samples
        sum_samples = number_samples_left + number_samples_right
        left_cost = self.impurity_measure.compute_impurity(left_dataset)
        right_cost = self.impurity_measure.compute_impurity(right_dataset)
        score= (number_samples_left / sum_samples) * left_cost+(number_samples_right / sum_samples) * right_cost
        return score