"""
Authors: Joaquín Flores Ruiz <jofloru023@gmail.com>
         Joan Carles Montero Jiménez <joan.carles.montero@gmail.com>
         Félix Fernández Peñafiel <felix2003fp@gmail.com>
"""


import numpy as np
from abc import ABC, abstractmethod
import time
from multiprocessing import Pool
from tqdm import tqdm
import logging.config
from numpy import ndarray
from Dataset import Dataset
from Node import Parent
from Visitor import PrinterTree, FeatureImportance
from Node import Node

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("RandomForest")


class RandomForest(ABC):
    # It follows the template method pattern
    # https://refactoring.guru/design-patterns/template-method
    """
        Random forest is a set of individual decision trees, each one trained
        with a sample slightly changed because of a random intervention.
        That makes the different decision trees have little differences,
        which allows us to make some predictions, to classify data or even
        detect anomalies in our Dataset.

        Attributes
        ----------
        impurity_measure : ImpurityMeasures
            Impurity Measure class instanced from the name_impurity attribute
            in the _make_impurity function. It is used in _cart_cost to
            calculate the best split.
        """
    def __init__(self, num_trees: int, min_size: int, max_depth: int,
                 ratio_samples: float, num_random_features: int,
                 name_impurity: str, multiprocessing: bool, extra_trees: bool):
        """
        Parameters
        ----------
        num_trees : int
            Number of decision trees

        min_size : int
            Minimum size of samples required to split.

        max_depth : int
            Maximum depth of the tree.

        ratio_samples : float
            Rate of the samples we will use from the "train dataset" to fit
            our trees.

        num_random_features : int
            Number of the features that will be used to fit our trees from the
            "train dataset"

        name_impurity : str
            The name of the impurity pattern used to estimate the cost
            of the split.

        multiprocessing : bool
            The parameter that points out if we will execute the program using
            multiprocessing (in the fit and predict function)

        extra_trees : bool
            The parameter that points out if we will execute the program using
            the extra trees procedure that consists in creating many more trees
            and splitting the datasets in a more simple way (only in the fit
            function).
        """
        self._num_trees = num_trees
        self._min_size = min_size
        self._max_depth = max_depth
        self._ratio_samples = ratio_samples
        self._num_random_features = num_random_features
        self._multiprocessing = multiprocessing
        self._extra_trees = extra_trees
        self._impurity_measure = self._make_impurity(name_impurity)

    def feature_importance(self) -> dict:
        """
        It's the visitor auxiliary function to go through the feature split
        indexes of all parent nodes to know their importance. Works in a
        recursive way.
        """
        feat_imp_visitor = FeatureImportance()
        for tree in self.decision_trees:
            tree.accept_visitor(feat_imp_visitor)
        return feat_imp_visitor.occurrences

    def print_trees(self) -> None:
        """
        It's the visitor auxiliary function to print all the generated trees.
        It operates in a recursive way, and it shows parents and leafs in
        different layers, depending on its depth.
        """
        tree_printer = PrinterTree()
        for tree in self.decision_trees:
            tree.accept_visitor(tree_printer)

    @abstractmethod
    def _make_impurity(self, name_impurity: str) :
        """
        Abstract function to instance the Impurity Measure.

        Parameters
        ----------
        name_impurity : str
            The name of the impurity pattern used to estimate the cost
            of the split.
        """
        pass

    @abstractmethod
    def _combine_predictions(self, predictions: list):
        """
        Abstract function to combine the list of predictions, entered by
        parameters.

        Parameters
        ----------
        predictions : list of floats
            The results of the final decisions of each tree.
        """
        pass

    def predict(self, X: ndarray) -> ndarray:
        """
        The auxiliary function to get the predictions with multiprocessing
        or not depending on the "multiprocessing" parameter.

        Parameters
        ----------
        X : array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample

        Returns
        -------
        predictions : array of floats
            The results of the final decisions of each tree.
        """
        if self._multiprocessing:
            predictions = self._predict_multiprocessing(X)
        else:
            predictions = self._predict_standard(X)
        return predictions

    def _predict_standard(self, X: ndarray) -> ndarray:
        """
        The function to get the decisions of the predictions of each tree, done
        through a recursive algorithm that calls the predict function of each
        Node (that ends up being the "label" attribute of the Leaf). This is
        done with the _combine_predictions function.

        Parameters
        ----------
        X: array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample

        Returns
        -------
        predictions : array of floats
            The results of the final decisions of each tree.
        """
        y_pred = []
        t1 = time.time()
        for x in tqdm(X, total=len(X), desc="Predict progress: "):
            predictions = [root.predict(x) for root in self.decision_trees]
            # majority voting
            y_pred.append(self._combine_predictions(predictions))
        t2 = time.time()
        logger.info('Predict lasted {} seconds'.format(t2 - t1))
        return np.array(y_pred)

    def _target_predict(self, x: ndarray) -> float:
        """
        An auxiliary multiprocessing function required by
        multiprocessing.Pool.map() in order to indicate what each core will
        have to do. In this case, in a recursive way, get the final label
        value of each Leaf with the _combine_predictions function.

        Parameters
        ----------
        x : array of floats
            representing the vector of features of one sample

        Returns
        -------
        prediction : array of floats
            The result of the final decisions of every tree, of an only sample
        """
        predictions = [root.predict(x) for root in self.decision_trees]
        prediction = self._combine_predictions(predictions)
        logger.debug("Prediction done, result: {}".format(prediction))
        return prediction

    def _predict_multiprocessing(self, X: ndarray) \
            -> ndarray:
        """
        The multiprocessing predict function that manages each core and tells
        them what to do. In this case to get the predictions through the target
        function _target_predict.

        Parameters
        ----------
        X: array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample

        Returns
        -------
        predictions : array of floats
            The results of the final decisions of each tree.
        """
        t1 = time.time()
        with Pool() as pool:
            y_pred = pool.map(self._target_predict,
                              tqdm(X, total=len(X),
                                   desc="Multiprocessing predict progress: "))
        t2 = time.time()
        logger.info('Multiprocessing predict lasted {} seconds'.
                    format(t2 - t1))
        return np.array(y_pred)

    def fit(self, X: ndarray, y: ndarray) -> None:  # train
        """
        The auxiliary function to fit our random forest trees with
        multiprocessing or not depending on the "multiprocessing" parameter

        Parameters
        ----------
        X: array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample
        y: array of length num_samples of int representing the labels or class
            of the samples
        """
        # its own responsibilities
        dataset = Dataset(X, y)
        if self._multiprocessing:
            self._make_decision_trees_multiprocessing(dataset)
        else:
            self._make_decision_trees_standard(dataset)

    def _make_decision_trees_standard(self, dataset: Dataset) -> None:
        """
        The recursive function that without multiprocessing creates the initial
        Nodes with depth 1 and add them into the decision_trees attribute. It
        uses a recursive algorithm calling the _make_node function with each
        initial Node of each tree.

        Parameters
        ----------
        The training dataset (X,y) used to make the subsets
        """
        self.decision_trees = []
        t1 = time.time()
        for _ in tqdm(range(self._num_trees),
                      total=self._num_trees, desc="Fit progress: "):
            # sample a subset of the dataset with replacement
            # using np.random.choice()
            # to get the indices of rows in X and y
            subset = dataset.random_sampling(self._ratio_samples)
            tree = self._make_node(subset, 1)  # the root of the decision tree
            self.decision_trees.append(tree)
        t2 = time.time()
        logger.info('Fit lasted {} seconds'.format(t2 - t1))

    def _target_fit(self, dataset: Dataset, nproc: int) -> Node:
        """
        An auxiliary multiprocessing function required by
        multiprocessing.Pool.map() in order to indicate what each core will
        have to do. In this case, in a recursive way, create the tree's Nodes
        in order to get the decision trees. To do this, it calls the _make_node
        function with the initial Node of a tree.

        Parameters
        ----------
        The training dataset (X,y) used to make the subsets
        nproc : int
            The number of process (only used in logging)

        Returns
        -------
        The root of the trained tree
        """
        logger.debug('process {} starts'.format(nproc))
        subset = dataset.random_sampling(self._ratio_samples)
        tree = self._make_node(subset, 1)
        logger.debug('process {} ends'.format(nproc))
        return tree

    def _make_decision_trees_multiprocessing(self, dataset: Dataset) -> None:
        """
        The multiprocessing fit function that manages each core and tells
        them what to do. In this case to create the decision trees through the
        target function _target_fit.

        Parameters
        ----------
        The training dataset (X,y) used to make the subsets
        """
        t1 = time.time()
        with Pool() as pool:
            iterable = [(dataset, number_process)
                        for number_process in range(self._num_trees)]
            self.decision_trees = \
                pool.starmap(self._target_fit,
                             tqdm(iterable, total=len(iterable),
                                  desc="Multiprocessing fit progress: "))
        t2 = time.time()
        logger.info('Multiprocessing fit lasted {} seconds'.format(t2 - t1))

    def _make_node(self, dataset: Dataset, depth: int) -> Node:
        """
        A recursive function that always returns a Node. Depending on some
        parameters, it returns a simple Leaf, of it calls _make_parent_or_leaf.
        If that happens, depending on the split done, it will return another
        Parent or a Leaf Node.

        Parameters
        ----------
        The random subset (X,y)

        depth : int
            Depth of the Node

        Returns
        -------
        A Node class that it can be a Leaf or Parent
        """
        if depth == self._max_depth \
                or dataset.num_samples <= self._min_size \
                or len(np.unique(dataset.y)) == 1:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset, depth)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node

    @abstractmethod
    def _make_leaf(self, dataset: Dataset, depth: int):
        """
        It returns a Leaf with a certain Leaf label.

        Parameters
        ----------
        The split dataset (X,y) created  by _best_split and asserts the
            condition
        depth : int
            Depth of the leaf
        """
        pass

    def _make_parent_or_leaf(self, dataset: Dataset, depth: int) -> Node:
        """
        This function generates a random subset of the Dataset introduced, and
        it splits it though _best_split. Depending on the results, it returns a
        Parent or a Leaf, both Nodes. It is a recursive function too.

        Parameters
        ----------
        The split dataset (X,y) created by _best_split
        depth : int
            Depth of the parent or leaf

        Returns
        -------
        A Node class that it can be a Leaf or Parent
        """
        # select a random subset of features, to make trees more diverse
        idx_features = np.random.choice(range(dataset.num_features),
                                        self._num_random_features,
                                        replace=False)

        best_feature_index, best_value, minimum_cost, best_split \
            = self._best_split(idx_features, dataset)
        left_dataset, right_dataset = best_split
        assert left_dataset.num_samples > 0 or right_dataset.num_samples > 0
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            # this is a special case : dataset has samples of at least two
            # classes but the best split is moving
            # all samples to the left or right
            # dataset and none to the other,
            # so we make a leaf instead of a parent
            return self._make_leaf(dataset, depth)
        else:
            node = Parent(best_feature_index, best_value)
            node.left_child = self._make_node(left_dataset, depth + 1)
            node.right_child = self._make_node(right_dataset, depth + 1)
            return node

    def _best_split(self, idx_features: ndarray, dataset: Dataset) \
            -> tuple:
        """
        The function that using _cart_cost, calculates the best dataset
        split, by going through all the index features and all the values and
        trying every combination (or only a random value of values if "extra
        trees"=True).

        Parameters
        ----------
        idx_features : array of ints
            Random subset of features
        dataset: The split dataset (X,y) created by _best_split

        Returns
        -------
        object: made up of the best pair (best_feature_index, best_value) that
            minimizes the cost function (minimum_cost), and best_split for
            the samples of the Dataset
        """
        # find the best pair (feature, value) by exploring all possible pairs
        best_feature_index, best_value, minimum_cost, best_split \
            = np.Inf, np.Inf, np.Inf, None
        if self._extra_trees:
            for idx in idx_features:
                max_ = np.amax(dataset.X[:, idx])
                min_ = np.amin(dataset.X[:, idx])
                val = np.random.uniform(min_, max_)
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._cart_cost(left_dataset, right_dataset)  # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_value, minimum_cost, best_split \
                        = idx, val, cost, [left_dataset, right_dataset]
        else:
            for idx in idx_features:
                values = np.unique(dataset.X[:, idx])
                for val in values:
                    left_dataset, right_dataset = dataset.split(idx, val)
                    # J(k,v)
                    cost = self._cart_cost(left_dataset, right_dataset)
                    if cost < minimum_cost:
                        best_feature_index, best_value, \
                            minimum_cost, best_split \
                            = idx, val, cost, \
                            [left_dataset, right_dataset]

        return best_feature_index, best_value, minimum_cost, best_split

    def _cart_cost(self, left_dataset: Dataset, right_dataset: Dataset) \
            -> float:
        """
        The function that returns the punctuation of the division of two
        datasets entered by parameters. It calculates the value of each of them
        with the impurity measure function chosen and later on it returns a
        score of the division taking into consideration the importance of each
        one (its amount of samples).

        Parameters
        ----------
        left_dataset: samples of the Dataset that are lower than the "split
            value"

        right_dataset: samples of the Dataset higher than the "split value"

        Returns
        -------
        score: an indicator which shows how good are the Impurity Measures.
            The lower the score, the better the Impurity Measure
        """
        number_samples_left = left_dataset.num_samples
        number_samples_right = right_dataset.num_samples
        sum_samples = number_samples_left + number_samples_right
        left_cost = self._impurity_measure.compute_impurity(left_dataset)
        right_cost = self._impurity_measure.compute_impurity(right_dataset)
        score = (number_samples_left / sum_samples) * left_cost \
            + (number_samples_right / sum_samples) * right_cost
        return score
