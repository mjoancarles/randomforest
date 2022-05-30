"""
Authors: Joaquín Flores Ruiz <jofloru023@gmail.com>
         Joan Carles Montero Jiménez <joan.carles.montero@gmail.com>
         Félix Fernández Peñafiel <felix2003fp@gmail.com>
"""

import numpy as np
from numpy import ndarray
import logging.config
from typing import List, Tuple
from RandomForest import RandomForest
from ImpurityMeasures import Gini, Entropy, SumSquareError, ImpurityMeasures
from Dataset import Dataset
from Node import Leaf

logging.config.fileConfig("logging.conf")
logger = logging.getLogger("RandomForest")


class RandomForestClassifier(RandomForest):
    """
    A Random Forest Classifier is a Random Forest class predictor that is
    trained with a sub-set of our samples. It uses the most frequent class
    predictions of all trees to predict the class type of a sample.

    Attributes
    ---------
    impurity_measure : ImpurityMeasures
        Impurity Measure class instanced from the name_impurity attribute
        in the _make_impurity function. It is used in _cart_cost to
        calculate the best split.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting
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
            The parameter that indicates if we will execute the program using
            the extra trees procedure that consists in creating many more trees
            and splitting the datasets in a more simple way (only in the fit
            function).
        """
        super().__init__(num_trees, min_size, max_depth,
                         ratio_samples, num_random_features, name_impurity,
                         multiprocessing, extra_trees)

    def _make_impurity(self, name_impurity: str) -> ImpurityMeasures:
        """
        It returns the Impurity Measure depending on a string parameter 
        entered.
        
        Parameters
        ----------
        name_impurity : str
            The name of the impurity pattern used to estimate the cost
            of the split.

        Returns
        -------
        A Impurity Measure class, either Gini or Entropy

        """
        if name_impurity.lower() == "gini":
            return Gini()
        elif name_impurity.lower() == "entropy":
            return Entropy()

    def _combine_predictions(self, predictions: ndarray) -> ndarray:
        """
        It returns the most repeated value of an array

        Parameters
        ----------
        predictions : array of floats
            The results of the final decisions of each tree.

        Returns
        -------
        The most frequent value of an array

        """
        return np.argmax(np.bincount(predictions))

    def _make_leaf(self, dataset: Dataset, depth: int) \
            -> Leaf:  # most frequent class in dataset
        """
        It returns a Leaf with a class number of its attribute, a result of
        the most frequent label of the dataset introduced.
        
        Parameters
        ----------
        The split dataset (X,y) created
        by _best_split and asserts the condition

        depth : int
            Depth of the leaf

        Returns
        -------
        A Leaf class

        """
        return Leaf(dataset.most_frequent_label())


class RandomForestRegressor(RandomForest):
    """
    A Random Forest Regressor is a Random Forest value predictor that is
    trained with a sub-set of our samples. It uses the mean predictions of
    every tree to predict the mean value of a sample.

    Attributes
    ---------
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
        super().__init__(num_trees, min_size, max_depth, ratio_samples,
                         num_random_features, name_impurity,
                         multiprocessing, extra_trees)

    def _make_impurity(self, name_impurity: str) -> SumSquareError:
        """
        It returns the Impurity Measure depending on a string parameter
        entered.

        Parameters
        ----------
         name_impurity : str
            The name of the impurity pattern used to estimate the cost
            of the split.

        Returns
        -------
        A Sum Square Error class

        """
        if name_impurity.lower() == "sum square error":
            return SumSquareError()

    def _combine_predictions(self, predictions: ndarray) \
            -> ndarray:
        """
        It returns the mean value of an array

        Parameters
        ----------
        predictions : array of floats
            The results of the final decisions of each tree.

        Returns
        -------
        mean value of an array

        """
        return np.mean(predictions)

    def _make_leaf(self, dataset: Dataset, depth: int) -> Leaf:
        """
        It returns a Leaf with the mean value of the dataset entered as the
        Leaf's attribute.

        Parameters
        ----------
        The split dataset (X,y) created
        by _best_split and asserts the condition

        depth : int
            Depth of the leaf

        Returns
        -------
            A Leaf class
        """
        return Leaf(dataset.mean_value())


class IsolationForest(RandomForest):
    """
    An Isolation Random Forest is a Random Forest algorithm trained with the
    whole set of samples that is able to give the anomaly score of each of them
    in order to detect which of them are anomalies.

    Attributes
    ---------
    train_size : int
        number of samples in the X_train dataset

    test_size : int
        number of samples in the X_test dataset

    """

    def __init__(self, num_trees: int, ratio_samples: float,
                 multiprocessing: bool):
        """
        Parameters
        ----------
        num_trees : int
            Number of decision trees

        ratio_samples : float
            Rate of the samples we will use from the "train dataset" to fit
            our trees.

        multiprocessing : bool
            The parameter that points out if we will execute the program using
            multiprocessing (in the fit and predict functions)
        """
        self._train_size = 0
        self._test_size = 0
        super().__init__(num_trees,
                         min_size=1,
                         max_depth=1,
                         ratio_samples=ratio_samples,
                         num_random_features=1,
                         name_impurity="",
                         multiprocessing=multiprocessing,
                         extra_trees=False)

    def fit(self, X: ndarray, y: ndarray = None) -> None:  # train
        """
        The random forest adapted function of an isolation forest, that does
        not receive a y, and, in order to solve it, we create a random one.
        Also, it defines the attributes _train_size (used in
        _combine_predictions) and _max_depth (used in _make_node). Once done,
        we call the original fit function.

        Parameters
        ----------
        X: array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample
        y: array of length num_samples of int representing the labels or class
            of the samples
        """
        self._train_size = len(X)
        y = np.random.rand(len(X))
        self._max_depth = int(np.log2(len(X)))
        super().fit(X, y)

    def predict(self, X: ndarray) -> ndarray:
        """
        The random forest adapted function of an isolation forest, that defines
        the necessary attribute _test_size used in _combine_predictions and
        that calls the original predict function.

        X: array of dimensions (num_samples, num_features) of float
            representing the vector of features of each sample

        Returns
        -------
        predictions : array of floats
            The results of the final decisions of Random Forest.

        """
        self._test_size = len(X)
        y_pred = super().predict(X)
        return y_pred

    def _make_impurity(self, name_impurity: str) -> None:
        """
        Empty function because of the lack of use of the _cart_cost in
        _best_split

        Parameters
        ----------
        name_impurity : str
            The name of the impurity pattern used to estimate the cost
            of the split.
        """
        pass

    def _combine_predictions(self, predictions: ndarray) -> float:
        """
        It returns the score between 0 and 1 applying the normalized anomaly
        formula, that indicates the anomaly grade of a sample.

        Parameters
        ----------
        predictions : array of floats
            The results of the final decisions of each tree.predictions

        Returns
        -------
        score : float
            Anomaly score


        """
        # predictions are the depths h(x) in the decision trees for a given x
        ehx = np.mean(predictions)  # mean depth
        cn = 2 * (np.log(self._train_size - 1) + 0.57721) \
            - 2 * (self._train_size - 1) / float(self._test_size)
        score = 2 ** (-ehx / cn)
        return score

    def _make_leaf(self, dataset: Dataset, depth: int) -> Leaf:
        """
        It returns a Leaf with the depth entered as a parameter as its
        attribute.

        Parameters
        ----------
        The split dataset (X,y) created
        by _best_split and asserts the condition

        depth : int
            Depth of the leaf

        Returns
        -------
        A Leaf class

        """
        return Leaf(depth)

    def _best_split(self, idx_features: ndarray, dataset: Dataset) \
            -> tuple:
        """
        The redefined _best_split function now splits the dataset with a random
        feature index and a random value of values (without using _cart_cost).

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
        max_idx = np.amax(idx_features)
        min_idx = np.amin(idx_features)
        idx = int(np.random.uniform(min_idx, max_idx))
        max_value = np.amax(dataset.X[:, idx])
        min_value = np.amin(dataset.X[:, idx])
        val = np.random.uniform(min_value, max_value)
        left_dataset, right_dataset = dataset.split(idx, val)
        best_split = [left_dataset, right_dataset]
        return idx, val, 0, best_split
