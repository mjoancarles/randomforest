import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
from RandomForestClassifier import RandomForestClassifier

#SONAR
def load_sonar():
    df = pd.read_csv('sonar.all-data',header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y=='M').astype(int) # M = mine, R = rock
    return X, y
X, y = load_sonar()
idx_rocks = y==0
idx_mines = y==1
plt.close('all')
plt.figure(), plt.plot(X[idx_rocks].T,'b'), plt.title('all samples of class rock')
plt.figure(), plt.plot(X[idx_mines].T,'r'), plt.title('all samples of class mine')


#IRIS DATASET
iris = sklearn.datasets.load_iris()
print(iris.DESCR)
X, y = iris.data, iris.target
# X 150 x 4, y 150 numpy arrays

ratio_train, ratio_test = 0.7, 0.3
# 70% train, 30% test
num_samples, num_features = X.shape
# 150, 4
idx = np.random.permutation(range(num_samples))
# shuffle {0,1, ... 149} because samples come sorted by class!
num_samples_train = int(num_samples*ratio_train)
num_samples_test = int(num_samples*ratio_test)
idx_train = idx[:num_samples_train]
idx_test = idx[num_samples_train : num_samples_train+num_samples_test]
X_train, y_train = X[idx_train], y[idx_train]
X_test, y_test = X[idx_test], y[idx_test]

# Hyperparameters
max_depth = 10 # maximum number of levels of a decision tree
min_size_split = 5 # if less, do not split a node
ratio_samples = 0.7 # sampling with replacement
num_trees = 10 # number of decision trees
num_random_features = int(np.sqrt(num_features))
# number of features to consider at each node when looking for the best split
criterion = 'gini' # 'gini' or 'entropy'
rf = RandomForestClassifier(num_trees, max_depth, min_size_split, ratio_samples, num_random_features, criterion)

# train = make the decision trees
rf.fit(X_train, y_train)
# classification

ypred = rf.predict(X_test)
# compute accuracy
num_samples_test = len(y_test)
num_correct_predictions = np.sum(ypred == y_test)
accuracy = num_correct_predictions/float(num_samples_test)
print('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))