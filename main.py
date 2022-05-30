import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pickle
import sklearn.datasets
import logging.config
from numpy import ndarray
from typing import Tuple
from RandomForest_classes import RandomForestClassifier, \
    RandomForestRegressor, IsolationForest


logging.config.fileConfig("logging.conf")
logger = logging.getLogger("main")


def basic_configuration(X: ndarray, y: ndarray) \
        -> tuple:
    ratio_train, ratio_test = 0.7, 0.3
    # 70% train, 30% test
    num_samples, num_features = X.shape
    # 150, 4
    idx = np.random.permutation(range(num_samples))
    # shuffle {0,1, ... 149} because samples come sorted by class!
    num_samples_train = int(num_samples * ratio_train)
    num_samples_test = int(num_samples * ratio_test)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train: num_samples_train + num_samples_test]
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_train, y_train, X_test, y_test


def load_mnist() -> tuple:
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    X_train, X_test = mnist["training_images"], mnist["test_images"]
    y_train, y_test = mnist["training_labels"], mnist["test_labels"]
    return X_train, y_train, X_test, y_test


def anomaly_detection(dataset_name: str) -> None:
    if dataset_name == "mnist":
        digit = 0
        save_images = True
        np.random.seed(1234)  # change this to get other outliers
        X_train, y_train, X_test, y_test = load_mnist()

        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        idx_digit = np.where(y == digit)[0]
        X = X[idx_digit]
        downsample = 2
        X2 = np.reshape(X, (len(X), 28, 28))[:, ::downsample, ::downsample]
        # from n vectors of 784 to n arrays 28 x 28
        X2 = np.reshape(X2, (len(X2), 28 * 28 // downsample ** 2))

        # from arrays 28 x 28 to 14 x 14 and back to vector, now of length 196
        iso = IsolationForest(num_trees=200,
                              ratio_samples=0.5,
                              multiprocessing=False)

        iso.fit(X2)
        logger.debug('Random Forest trained')
        scores = iso.predict(X2)
        logger.debug('Predictions done')
        # histogram of the scores to check very few have a score close to 1
        plt.close('all')
        plt.figure()
        plt.hist(scores, bins=100)
        plt.xlabel('score'), plt.ylabel('frequency'),
        plt.title('digit {}'.format(digit))
        # find the 0.2% of samples with the highest score
        # that we'll consider anomalies
        percent_outliers = 0.2
        num_digits = len(X)
        num_outliers = int(np.ceil(num_digits * percent_outliers / 100.))
        sorted_scores = np.sort(scores)
        thr = sorted_scores[-num_outliers]
        idx_anomalies = np.where(scores > thr)[0]
        # plot and maybe save the images
        num_ima = 1
        ima=0
        for ii in idx_anomalies:
            ima = np.reshape(X[ii], (28, 28))
            plt.figure(), plt.imshow(ima, cmap=plt.cm.gray)
        if save_images:
            plt.imsave('{}_{}.png'.format(digit, num_ima), ima)
        num_ima += 1
        plt.show(block=False)


def print_trees(dataset: str) -> None:
    filename = "rf_" + dataset + ".pkl"
    with open(filename, 'rb') as f:
        rf = pickle.load(f)
    logger.info("random forest imported from pickle file " + filename)
    rf.print_trees()


def feature_importance(dataset_name: str) -> None:
    filename = "rf_" + dataset + ".pkl"
    with open(filename, 'rb') as f:
        rf = pickle.load(f)
    logger.info("random forest imported from pickle file " + filename)
    if dataset_name == "sonar":
        occurrences = rf.feature_importance()  # a dictionary
        counts = np.array(list(occurrences.items()))
        plt.figure()
        plt.bar(counts[:, 0], counts[:, 1])
        plt.xlabel('feature')
        plt.ylabel('occurrences')
        plt.title('Sonar feature importance\n{} trees'.format(rf._num_trees))
        plt.show()
    elif dataset_name == "iris":
        occurrences = rf.feature_importance()
        print('Iris occurrences for {} trees'.format(rf._num_trees))
        print(occurrences)
    elif dataset_name == "mnist":
        occurrences = rf.feature_importance()
        ima = np.zeros(28 * 28)
        for k in occurrences.keys():
            ima[k] = occurrences[k]
        plt.figure()
        plt.imshow(np.reshape(ima, (28, 28)))
        plt.colorbar()
        plt.title('Feature importance MNIST')
        plt.show()


def iris_test() -> None:
    iris = sklearn.datasets.load_iris()
    # print(iris.DESCR)
    X, y = iris.data, iris.target
    # X 150 x 4, y 150 numpy arrays
    X_train, y_train, X_test, y_test = basic_configuration(X, y)
    logger.info('Iris Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 80  # number of decision trees
    num_random_features = int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    # 'gini', 'entropy'
    criterion = 'gini'
    logger.debug('Hyper-parameters defined')

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth,
                                ratio_samples, num_random_features, criterion,
                                multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions / float(num_samples_test)
    logger.info('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
    t2 = time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2 - t1))

    with open('rf_iris.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save rf object, so we won't need to train again


def sonar_test() -> None:
    df = pd.read_csv('sonar.all-data', header=None)
    X = df[df.columns[:-1]].to_numpy()
    y = df[df.columns[-1]].to_numpy(dtype=str)
    y = (y == 'M').astype(int)  # M = mine, R = rock
    idx_rocks = y == 0
    idx_mines = y == 1
    plt.close('all')
    plt.figure(), plt.plot(X[idx_rocks].T, 'b'), \
        plt.title('all samples of class rock')
    plt.figure(), plt.plot(X[idx_mines].T, 'r'), \
        plt.title('all samples of class mine')
    # plt.show()
    X_train, y_train, X_test, y_test = basic_configuration(X, y)
    logger.info('Sonar Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 80  # number of decision trees
    num_random_features = int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = "gini"  # 'gini', 'entropy'
    logger.debug('Hyper-parameters defined')

    rf = RandomForestClassifier(
        num_trees, min_size_split, max_depth,
        ratio_samples, num_random_features,
        criterion, multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions / float(num_samples_test)
    logger.info('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
    t2 = time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2 - t1))

    with open('rf_sonar.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save rf object, so we won't need to train again


def mnist_test() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    n_sample = 0
    plt.close('all')
    plt.figure()
    for i in range(1):
        for j in range(20):
            n_sample = 20 * i + j

    plt.subplot(10, 20, n_sample + 1)
    plt.imshow(np.reshape(X_train[n_sample], (28, 28)),
               interpolation='nearest', cmap=plt.cm.gray)
    plt.title(str(y_train[n_sample]), fontsize=8)
    plt.axis('off')
    plt.show()
    logger.info('MNIST Dataset imported')

    # np.random.seed(123)
    max_depth = 10  # maximum number of levels of a decision tree
    min_size_split = 20  # if less, do not split a node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 20  # number of decision trees
    num_random_features = int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = 'gini'  # 'gini', 'entropy'
    logger.debug('Hyper-parameters defined')

    rf = RandomForestClassifier(num_trees, min_size_split, max_depth,
                                ratio_samples, num_random_features, criterion,
                                multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')
    # compute accuracy
    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions / float(num_samples_test)
    logger.info('accuracy {} %'.format(100 * np.round(accuracy, decimals=2)))
    t2 = time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2 - t1))

    with open('rf_mnist.pkl', 'wb') as f:
        pickle.dump(rf, f)  # save rf object, so we won't need to train again


def daily_min_temp_test() -> None:
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
    num_trees = 60  # number of decision trees
    num_random_features = int(np.sqrt(X_train.shape[1]))
    multiprocess = True
    extra_trees = True
    criterion = "sum square error"  # "sum square error"
    logger.debug('Hyper-parameters defined')

    rf = RandomForestRegressor(num_trees, min_size_split, max_depth,
                               ratio_samples, num_random_features, criterion,
                               multiprocess, extra_trees)
    rf.fit(X_train, y_train)
    logger.debug('Random Forest trained')

    ypred = rf.predict(X_test)
    logger.debug('Predictions done')

    t2 = time.time()
    logger.info('The whole program has lasted {} seconds'.format(t2 - t1))
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
        pickle.dump(rf, f)  # save rf object, so we won't need to train again


def credit_card_fraud_test() -> None:
    df = pd.read_csv('creditcard_10K.csv', header=None)

    X = np.array(df)
    X = X[:, 1:]  # remove first feature
    y = X[:, -1]
    X = X[:, :-1]
    del df
    num_samples = len(X)
    print('{} number of samples'.format(num_samples))
    np.random.seed(123)  # to get replicable results
    idx = np.random.permutation(num_samples)
    X = X[idx]  # shuffle
    y = y[idx]
    print('{} samples, {} outliers, {} % '.format(len(y), y.sum(),
                                                  np.round(
                                                      100 * y.sum() / len(y),
                                                      decimals=3)))

    num_trees = 500
    ratio_samples = 0.1
    logger.debug("Hyper-parameters defined")
    iso = IsolationForest(num_trees, ratio_samples, multiprocessing=True)
    # with multiprocessing=False similar time and results
    iso.fit(X)
    logger.debug("Isolation Forest trained")
    scores = iso.predict(X)
    logger.debug("Predictions done")
    plt.figure(), plt.hist(scores, bins=100)
    plt.title('histogram of scores')
    percent_anomalies = 0.5
    num_anomalies = int(percent_anomalies * num_samples / 100.)
    idx = np.argsort(scores)
    idx_predicted_anomalies = idx[-num_anomalies:]
    precision = y[idx_predicted_anomalies].sum() / num_anomalies
    print('precision for {} % anomalies : {} %'
          .format(percent_anomalies, 100 * precision))
    recall = y[idx_predicted_anomalies].sum() / y.sum()
    print('recall for {} % anomalies : {} %'
          .format(percent_anomalies, 100 * recall))


def synthetic_dataset_test() -> None:
    rng = np.random.RandomState(42)
    X = 0.3 * rng.randn(100, 2)  # synthetic dataset
    X_train = np.r_[X + 2, X - 2]
    X = 0.3 * rng.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Xgrid = np.c_[xx.ravel(), yy.ravel()]  # where to compute the score
    num_trees = 100
    ratio_samples = 0.5
    multiprocessing = True

    logger.debug("Hyper-parameters defined")
    iso = IsolationForest(num_trees, ratio_samples,
                          multiprocessing)
    iso.fit(X_train)
    logger.debug("Isolation Forest trained")
    scores = iso.predict(Xgrid)
    logger.debug("Predictions done")
    scores = scores.reshape(xx.shape)
    plt.title("IsolationForest")
    m = plt.contourf(xx, yy, scores, cmap=plt.cm.Blues_r)
    plt.colorbar(m)
    b1 = plt.scatter(X_train[:, 0], X_train[:, 1],
                     c="white", s=20, edgecolor="k")
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1],
                     c="green", s=20, edgecolor="k")
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1],
                    c="red", s=20, edgecolor="k")
    plt.axis("tight"), plt.xlim((-5, 5)), plt.ylim((-5, 5))
    plt.legend([b1, b2, c], ["training observations",
                             "new regular observations",
                             "new abnormal observations"], loc="upper left")
    plt.show()


if __name__ == '__main__':
    t1 = time.time()
    if len(sys.argv) != 3:
        logger.critical("Invalid number of arguments, "
                        "please enter one to select the dataset")

    dataset = sys.argv[1].lower()

    importance = test = anomaly = printer = False

    if sys.argv[2].lower() == "print":
        logger.debug("printer variable set to true")
        printer = True
    elif sys.argv[2].lower() == "importance":
        logger.debug("importance variable set to true")
        importance = True
    elif sys.argv[2].lower() == "test":
        logger.debug("test variable set to true")
        test = True
    elif sys.argv[2].lower() == "anomaly":
        logger.debug("anomaly variable set to true")
        anomaly = True

    if test:
        if dataset == "iris":
            logger.debug("iris_test function called")
            iris_test()
        elif dataset == "sonar":
            logger.debug("sonar_test function called")
            sonar_test()
        elif dataset == "mnist":
            logger.debug("mnist_test function called")
            mnist_test()
        elif dataset == "daily_min_temp":
            logger.debug("daily_min_temp_test function called")
            daily_min_temp_test()
        elif dataset == "synthetic_dataset":
            logger.debug("synthetic_dataset_test function called")
            synthetic_dataset_test()
        elif dataset == "credit_card_fraud":
            logger.debug("credit_card_fraud_test function called")
            credit_card_fraud_test()

    elif printer:
        logger.debug("print function called")
        print_trees(dataset)
    elif importance:
        logger.debug("feature importance function called")
        feature_importance(dataset)
    elif anomaly:
        logger.debug("anomaly detection function called")
        anomaly_detection(dataset)
