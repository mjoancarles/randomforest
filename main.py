import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import logging.config
import sklearn.datasets
from RandomForest_classes import *
import multiprocessing
import pickle
import os

logging.config.fileConfig("logging.conf")
logger=logging.getLogger("main")

def basic_configuration(X,y):
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

    return X_train, y_train, X_test, y_test

def print_trees(dataset):
    filename= "rf_"+dataset+".pkl"
    with open(filename, 'rb') as f:
        rf = pickle.load(f)
    logger.info("random forest imported from pickle file " + filename)
    rf.print_trees()

def feature_importance(dataset):
    filename = "rf_" + dataset + ".pkl"
    with open(filename, 'rb') as f:
        rf = pickle.load(f)
    logger.info("random forest imported from pickle file "+filename)
    if dataset=="sonar":
        occurrences = rf.feature_importance()  # a dictionary
        counts = np.array(list(occurrences.items()))
        plt.figure()
        plt.bar(counts[:, 0], counts[:, 1])
        plt.xlabel('feature')
        plt.ylabel('occurrences')
        plt.title('Sonar feature importance\n{} trees'.format(rf.num_trees))
        plt.show()
    elif dataset=="iris":
        occurrences = rf.feature_importance()
        print('Iris occurrences for {} trees'.format(rf.num_trees))
        print(occurrences)
    elif dataset=="mnist":
        occurrences = rf.feature_importance()
        ima = np.zeros(28 * 28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima, (28, 28)))
        plt.colorbar()
        plt.title('Feature importance MNIST')
        plt.show()

def iris_test():
    iris = sklearn.datasets.load_iris()
    #print(iris.DESCR)
    X, y = iris.data, iris.target
    # X 150 x 4, y 150 numpy arrays
    X_train, y_train, X_test, y_test = basic_configuration(X, y)
    logger.info('Iris Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 20  # number of decision trees
    multiprocessing.cpu_count() == 8
    num_random_features=int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = 'gini'  # 'gini', 'entropy'
    logger.debug('Hyperparameters defined')

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterion, multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    logger.info('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    t2=time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2-t1))

    with open('rf_iris.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save the rf object so we won't need to train again

def sonar_test():
    df = pd.read_csv('sonar.all-data', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y == 'M').astype(int)  # M = mine, R = rock
    idx_rocks = y == 0
    idx_mines = y == 1
    plt.close('all')
    plt.figure(), plt.plot(X[idx_rocks].T, 'b'), plt.title('all samples of class rock')
    plt.figure(), plt.plot(X[idx_mines].T, 'r'), plt.title('all samples of class mine')
    # plt.show()
    X_train, y_train, X_test, y_test = basic_configuration(X, y)
    logger.info('Sonar Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 20  # number of decision trees
    multiprocessing.cpu_count() == 8
    num_random_features=int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion= "gini" # 'gini', 'entropy'
    logger.debug('Hyperparameters defined')

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterion,multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    logger.info('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    t2=time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2-t1))

    with open('rf_sonar.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save the rf object so we won't need to train again

def mnist_test():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    X_train, X_test = mnist["training_images"], mnist["test_images"]
    y_train, y_test = mnist["training_labels"], mnist["test_labels"]

    plt.close('all')
    plt.figure()
    for i in range(1):
        for j in range(20):
            n_sample = 20 * i + j
    plt.subplot(10, 20, n_sample + 1)
    plt.imshow(np.reshape(X_train[n_sample], (28, 28)), interpolation='nearest', cmap=plt.cm.gray)
    plt.title(str(y_train[n_sample]), fontsize=8)
    plt.axis('off')
    plt.show()
    logger.info('MNIST Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 20  # number of decision trees
    multiprocessing.cpu_count() == 8
    num_random_features=int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = 'gini'  # 'gini', 'entropy'
    logger.debug('Hyperparameters defined')

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterion,multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions/float(num_samples_test)
    logger.info('accuracy {} %'.format(100*np.round(accuracy, decimals=2)))
    t2=time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2-t1))

    with open('rf_mnist.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save the rf object so we won't need to train again

def daily_min_temp_test():
    df = pd.read_csv("daily-min-temperatures.csv")
    # Minimum Daily Temperatures Dataset over 10 years (1981-1990)
    # in Melbourne, Australia. The units are in degrees Celsius.
    # These are the features to regress:
    day = pd.DatetimeIndex(df.Date).day.to_numpy()  # 1...31
    month = pd.DatetimeIndex(df.Date).month.to_numpy()  # 1...12
    year = pd.DatetimeIndex(df.Date).year.to_numpy()  # 1981...1999
    X = np.vstack([day, month, year]).T  # np array of 3 columns
    y = df.Temp.to_numpy()
    last_years_test = 1
    plt.figure()
    plt.plot(range(len(day)), y, '.-')
    plt.xlabel('day in 10 years'), plt.ylabel('min. daily temperature')
    idx = last_years_test * 365
    X_train = X[:-idx, :]  # first years
    X_test = X[-idx:]
    y_train = y[:-idx]  # last years
    y_test = y[-idx:]

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 20  # number of decision trees
    multiprocessing.cpu_count() == 8
    num_random_features = int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = "sum square error" # "sum square error"
    logger.debug('Hyperparameters defined')

    rf = RandomForestRegressor(num_trees, min_size_split, max_depth, ratio_samples, num_random_features, criterion, multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')

    t2=time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2-t1))
    last_years_test = 1
    plt.figure()
    x = range(last_years_test * 365)
    for t, y1, y2 in zip(x, y_test, ypred):
        plt.plot([t, t], [y1, y2], 'k-')
    plt.plot([x[0], x[0]], [y_test[0], ypred[0]], 'k-', label='error')
    plt.plot(x, y_test, 'g.', label='test')
    plt.plot(x, ypred, 'y.', label='prediction')
    plt.xlabel('day in last {} years'.format(last_years_test))
    plt.ylabel('min. daily temperature')
    plt.legend()
    errors = y_test - ypred
    rmse = np.sqrt(np.mean(errors ** 2))
    plt.title('root mean square error : {:.3f}'.format(rmse))
    plt.show()

    with open('rf_daily.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save the rf object so we won't need to train again

if  __name__ == '__main__':
    t1=time.time()
    if len(sys.argv) != 3:
        logger.critical("Invalid number of arguments, please enter one to select the dataset")

    dataset=sys.argv[1].lower()

    if sys.argv[2].lower()=="print":
        logger.debug("printer variable set to true")
        printer=True
        importance=test=False
    elif sys.argv[2].lower() == "importance":
        logger.debug("importance variable set to true")
        importance=True
        printer=test=False
    elif sys.argv[2].lower() == "test":
        logger.debug("test variable set to true")
        test = True
        importance=printer=False

    if test:
        if dataset=="iris":
            logger.debug("iris_test function called")
            iris_test()
        elif dataset=="sonar":
            logger.debug("sonar_test function called")
            sonar_test()
        elif dataset=="mnist":
            logger.debug("mnist_test function called")
            mnist_test()
        elif dataset=="daily_min_temp":
            logger.debug("daily_min_temp_test function called")
            daily_min_temp_test()
    elif printer:
        logger.debug("print function called")
        print_trees(dataset)
    elif importance:
        logger.debug("feature importance function called")
        feature_importance(dataset)