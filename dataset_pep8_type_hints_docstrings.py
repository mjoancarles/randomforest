#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 21:01:47 2021

@author: joan

I'm documenting this class with Docstrings *just as an example*. It's not
necessary doing it with Docstrings because it's a class used by fit() method
in RandomForest, not a class to be used by a programmer.
This is the Dataset class of the first milestone, only for classification,
no regression yet
"""
import numpy as np


class Dataset:
    """
    In classification, a dataset is a pair (X,y) where X is the matrix of
    samples and y their class or label. A sample is a vector of features. A
    label is a class number between 0...C-1, being C the number of classes.

    This class internally used by base abstract class RandomForest when building
    the decision trees. Its method fit(X,y) takes the pair X, y and makes a
    dataset with them that is successively split into left and right datasets
    each time it makes a new parent node of a decision tree.

    The class supports through its public methods the functionality needed by
    fit() that depends on the values of X and y :
    - sample a random subset of the dataset
    - split itself into left and right datasets according to a certain
      feature and threshold value for instance to optimize for the best feature
      and threshold value according to an impurity measure
    - get the probability distribution by classes in y
    - get the mode or most frequent class in y

    Attributes
    ----------
    num_samples : int
        the number of samples, equal to the number of rows of X and length of y

    num_features : int
        number of columns of X

    Examples
    --------
    >>> from dataset import Dataset
    >>> import numpy as np
    >>> ds = Dataset(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2],
                     dtype=np.int))
    """
    def __init__(self, X: np.array, y: np.array):
        """
        Parameters
        ----------
        X : array of shape (num_samples, num_features) of float representing
            the vector of features of each sample
        y : array of length num_samples of int representing the labels or class
            of the samples
        """

        # TODO: make attributes "private" by prefixing them with underscore,
        #   or better, add property getters
        self.X: np.array = X
        self.y: np.array = y
        self.num_samples: int
        self.num_features: int
        self.num_samples, self.num_features = X.shape

    def random_sampling(self, ratio: float, replace: bool = True) -> Dataset:
        """
        Parameters
        ----------
        ratio : float
            ratio over 1.0 of samples to draw

        replace : boolean, default True
            whether to draw samples with (default) or without replacement

        Returns
        -------
        A new dataset object with drawn rows of X and y
        """
        assert 0.0 < ratio <= 1.0
        size: int = int(self.num_samples * ratio)
        assert size > 0
        idx: float = np.random.choice(range(self.num_samples), size, replace)
        return Dataset(self.X[idx], self.y[idx])

    def split(self, index_feature: int, threshold: float) -> List[Dataset]:
        """
        Splits the current dataset in two according to a certain column of X and
        a threshold value.

        Parameters
        ----------
        index_feature : int
            number of column of X, $k$ in the equations

        threshold : float
            value to select rows of X and elements of y, $v$ in the equations

        Returns
        -------
        Two datasets, one made of the rows of X and y such that X[:,k] < v and
        the other X[:,k] >= v
        """
        idx_left: int = self.X[:, index_feature] < threshold
        idx_right: int = self.X[:, index_feature] >= threshold
        return Dataset(self.X[idx_left], self.y[idx_left]), \
            Dataset(self.X[idx_right], self.y[idx_right])

    def distribution(self) -> np.array:
        """
        Returns
        -------
        Probability distribution of classes in y, that is, the $p_c$ of the
        equations. It's just the frequency normalized to 1.
        Caution: np.bincount(z) returns an array of length equal to the maximum
        value in vector z, that is, the largest in y label + 1, which may be
        less than the number of classes of the problem. But it's ok for the
        formulas that use the distribution.
        """
        counts: np.array = np.bincount(self.y)
        return counts / float(np.sum(counts))

    def most_frequent_label(self) -> np.array:
        """
        Returns
        -------
        The mode of y
        """
        counts: np.array = np.bincount(self.y)
        return np.argmax(counts)
