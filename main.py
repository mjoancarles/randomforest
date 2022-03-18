import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import logging.config
import sklearn.datasets
from RandomForestClassifier import RandomForestClassifier
import multiprocessing
import pickle
import argparse

logging.config.fileConfig("logging.conf")
logger=logging.getLogger("main")

def dataset_selection(dataset):
    if dataset.lower()=="iris":
        # IRIS DATASET
        iris = sklearn.datasets.load_iris()
        print(iris.DESCR)
        X, y = iris.data, iris.target
        # X 150 x 4, y 150 numpy arrays
        logger.info('Iris Dataset imported')
    elif dataset.lower()=="sonar":
        # SONAR
        df = pd.read_csv('sonar.all-data', header=None)
        X = df[df.columns[:-1]].to_numpy()
        y = df[df.columns[-1]].to_numpy(dtype=str)
        y = (y == 'M').astype(int)  # M = mine, R = rock
        idx_rocks = y == 0
        idx_mines = y == 1
        plt.close('all')
        plt.figure(), plt.plot(X[idx_rocks].T, 'b'), plt.title('all samples of class rock')
        plt.figure(), plt.plot(X[idx_mines].T, 'r'), plt.title('all samples of class mine')
        logger.info('Sonar Dataset imported')
    elif dataset.lower()=="mnist":
        pass
        '''
        with open("mnist.pkl",'rb') as f:
            mnist=pickle.load(f)
        X = mnist["training_images"].concatenate(mnist["test_images"],axis=None)
        #X,y = mnist["training_images"].put(mnist["test_images"]),mnist["training_labels"].put(mnist["test_labels"])
        plt.close('all')
        plt.figure()
        for i in range(10):
            for j in range(20):
                n_sample = 20 * i + j
        plt.subplot(10, 20, n_sample + 1)
        plt.imshow(np.reshape(Xtrain[n_sample], (28, 28)),interpolation='nearest', cmap=plt.cm.gray)
        plt.title(str(ytrain[n_sample]), fontsize=8)
        plt.axis('off')
        logger.info('MNIST Dataset imported')
        '''
    else:
        logger.critical("The selection of the dataset was incorrecly made...")
    return X,y

if  __name__ == '__main__':
    t1=time.time()
    if len(sys.argv)!=2:
        logger.critical("Invalid number of arguments, please enter one to select the dataset")
    X,y = dataset_selection(sys.argv[1]) #sys.argv[0] is the path to the main file

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
    num_trees = 8000 # number of decision trees
    multiprocessing.cpu_count()==8
    num_random_features = int(np.sqrt(num_features))
    # number of features to consider at each node when looking for the best split
    criterion = 'entropy' # 'gini' or 'entropy'
    logger.debug('Hyperparameters defined')
    rf = RandomForestClassifier(num_trees, max_depth, min_size_split, ratio_samples, num_random_features, criterion)

    # train = make the decision trees
    rf.fit(X_train, y_train)
    logger.debug('Random Forest Classifier trained')
    # classification


    #ypred = rf.predict_multiprocessing(X_test)
    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    logger.info('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    t2=time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2-t1))