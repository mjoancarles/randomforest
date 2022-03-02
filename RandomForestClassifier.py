import numpy as np
from Dataset import Dataset
from Leaf import Leaf
from Parent import Parent


class RandomForestClassifier:
    def __init__(self, num_trees, min_size, max_depth, ratio_samples, num_random_features, criterion):
        self.num_trees=num_trees
        self.min_size=min_size
        self.max_depth=max_depth
        self.ratio_samples=ratio_samples
        self.num_random_features=num_random_features
        self.criterion=criterion

    def fit(self, X, y): #train
        # a pair (X,y) is a dataset, with its own responsibilities
        dataset = Dataset(X, y)
        self._make_decision_trees(dataset)

    def predict(self,X):
        ypred = []
        for x in X:
            predictions = [root.predict(x) for root in self.decision_trees]
            # majority voting
            counts = np.bincount(predictions)
            ypred.append(np.argmax(counts))
        return np.array(ypred)

    def _make_decision_trees(self, dataset):
        self.decision_trees = []
        for i in range(self.num_trees):
            # sample a subset of the dataset with replacement using np.random.choice()
            # to get the indices of rows in X and y
            subset = dataset.random_sampling(self.ratio_samples)
            tree = self._make_node(subset,1) # the root of the decision tree
            self.decision_trees.append(tree)

    def _make_node(self, dataset, depth):
        if depth == self.max_depth or dataset.num_samples <= self.min_size or len(np.unique(dataset.y)) == 1:
            # last condition is true if all samples belong to the same class
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)
        return node

    def _make_leaf(self, dataset):
        label = dataset.most_frequent_label() # most frequent class in dataset
        return Leaf(label)

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
        for idx in idx_features:
            values = np.unique(dataset.X[:,idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                cost = self._CART_cost(left_dataset, right_dataset)  # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_value, minimum_cost, best_split = idx, val, cost, [left_dataset, right_dataset]
        return best_feature_index, best_value, minimum_cost, best_split

    def _CART_cost(self, left_dataset, right_dataset):
        #the J(k,v) equation in the slides, using Gini
        samples_left=left_dataset.num_samples
        samples_right=right_dataset.num_samples
        sum_samples= samples_left+samples_right
        cost=(samples_left/sum_samples)*self._gini(left_dataset)+(samples_right/sum_samples)*self._gini(right_dataset)
        return cost

    def _gini(self, dataset):
        count=np.bincount(dataset.y)
        sum_gini=sum(list(map(lambda x: (x/dataset.num_samples)**2, count)))
        return 1-sum_gini

