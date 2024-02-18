import numpy as np
from torchvision import datasets
from sklearn.datasets import fetch_california_housing


def get_MNIST_data(num_training=50000, num_validation=10000, num_test=10000):
    """
    Load the MNIST dataset from disk and perform preprocessing to prepare
    it for the classification. 
    """
    # Load the raw MNIST data
    train_data = datasets.MNIST('./data', train=True, download=True)
    test_data = datasets.MNIST('./data', train=False, download=True)

    X_train, y_train = np.array(train_data.data, dtype=float), np.array(
        train_data.targets, dtype=float)
    X_test, y_test = np.array(test_data.data, dtype=float), np.array(
        test_data.targets, dtype=float)

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_normalized_MNIST_data(X_train, X_val, X_test):
    # Normalize the data: subtract the mean image
    epsilon = 10 ** -6
    mean_image = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + epsilon

    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    return X_train / std , X_val / std , X_test / std

def get_california_housing_data(num_training=15640, num_validation=2500, num_test=2500):
    """
    Load the california housing dataset from disk and perform preprocessing to prepare
    it for the price prediction. 
    """

    # Load the raw california_housing data
    X_train, y_train = fetch_california_housing(return_X_y=True)

    # subsample the data
    mask = list(range(num_training + num_validation,
                      num_training + num_validation + num_test))
    X_test = X_train[mask]
    y_test = y_train[mask]
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_california_housing_normalized__data(X_train, X_val, X_test):
    # Normalize the data: subtract the mean array
    epsilon = 10 ** -6
    mean_array = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + epsilon

    X_train -= mean_array
    X_val -= mean_array
    X_test -= mean_array
    return X_train / std, X_val / std, X_test / std
