import random
import numpy as np
import matplotlib.pyplot as plt


def read_tsv(file_path):
    observations = open(file_path).read().strip().split('\n')
    observations = [[1] + list(map(float, obs.split())) for obs in observations]
    X = [obs[:-1] for obs in observations]
    y = [obs[-1] for obs in observations]
    return X, y


def batch_gdescent(X, y, alpha, w, epochs=500):
    alpha = alpha/len(X)
    for epoch in range(epochs):
        loss = y - X.dot(w)
        gradient = X.T.dot(loss) 
        w_old = w
        w = w + alpha * gradient
        if np.linalg.norm(w - w_old) / np.linalg.norm(w) < 0.0005:      # From Pierres code in order to break if batch
            break                                                       # gradient descent is done before epochs
    return w


def sgd(X, y, alpha, w, epochs=500):
    random.seed(0)
    idx = list(range(len(X)))
    for epoch in range(epochs):
        random.shuffle(idx)
        w_old = w
        for i in idx:
            loss = y[i] - X[i].dot(w)
            gradient = loss * X[i].reshape(-1, 1)
            w = w + alpha * gradient
            if np.linalg.norm(w - w_old) / np.linalg.norm(w) < 0.005:   # From Pierres code in order to break if batch
                break                                                   # gradient descent is done before epochs
    return w


def normalize(x_values, y_values):
    print("X-values: ", x_values)
    print("--------------------------------------------------------------------------")
    print("Y-values: ", np.array(y_values).T)
    x_values = [x / max(x_values) for x in x_values]
    y_values = [y / max(y_values) for y in y_values]
    print("############################## Normalized ################################")
    print("X-values: ", x_values)
    print("--------------------------------------------------------------------------")
    print("Y-values: ", y_values)
    return x_values, y_values
    

def numpy_array(x_values, y_values):
    return np.array(x_values), np.array([y_values]).T


def plot_bgd(x_values, y_values, alpha, epochs, title):
    w = np.zeros(x_values[1].shape).reshape((-1, 1))
    w = batch_gdescent(x_values, y_values, alpha, w, 500)
    plot_graph(x_values, y_values, alpha, epochs, title, w)
    return w


def plot_sgd(x_values, y_values, alpha, epochs, title):
    w = np.zeros(x_values.shape[1]).reshape((-1, 1))
    w = sgd(x_values, y_values, alpha, w, epochs)
    plot_graph(x_values, y_values, alpha, epochs, title, w)
    return w


def plot_graph(x_values, y_values, alpha, epochs, title, w):
    plt.scatter(x_values[:,1], y_values, c='b', marker='x', label="Values")
    x_axis = x_values[:,1]
    y_axis = []
    for i in range(len(x_values)):
        y_axis.append(w[1] + w[0] * x_values[i][1])
    fitted_values = np.polyfit(x_axis, y_axis, deg=1)
    plt.plot(fitted_values, '-', label="Regression line")
    plt.xlabel("Frequency of A's")
    plt.ylabel("Amount of letters")
    plt.legend()
    plt.title(title)
    plt.show()
