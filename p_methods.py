import numpy as np
import matplotlib.pyplot as plt


def looc(matrix, alpha):
    missclassified_eval = 0
    tries = 0
    for i in range(len(matrix)):
        evaluation_sample = matrix[i]
        training_set = np.delete(matrix, i, axis=0)
        w = [0.4,1.0,1.0]
        ######################## Training Weights #######################
        w, count = train_weights(training_set, w, alpha)
        tries += count
        _, right, count = evaluate(evaluation_sample, w)
        tries += count

        if not right: missclassified_eval += 1
        print(f"Prediction: {_} class: {evaluation_sample[3]}")
        print("Right: ", right)
        print("Missclassified count: ", missclassified_eval)
        # plot_perceptron(matrix, w, alpha, evaluation_sample, i)                       # Uncomment in order to plot every model
        print(f"########################### Evaluation ##########################")
        print(f"Missclassification: {missclassified_eval / tries}%")
    return w


def read_libsvm():
    dataset = open('/Users/sofiflink/Skola/Pågående Kurser/Artificiell Intelligens EDAP01/machine-learning/salammbo_a_binary.libsvm').read().strip().split('\n')
    observations = [dataset[i].split() for i in range(len(dataset))]
    y = [float(obs[0]) for obs in observations]
    X = [['0:1'] + obs[1:] for obs in observations]
    X = [list(map(lambda x: float(x.split(':')[1]), obs)) for obs in X]
    return X, y


def numpy_hstack(x_values, y_values):
    return np.hstack((np.array(x_values), np.array([y_values]).T))

    
def normalize(x_values, y_values):
    print("X-values: ", x_values)
    print("--------------------------------------------------------------------------")
    print("Y-values: ", np.array(y_values).T)
    x_values = [x / max(x_values) for x in x_values]
    y_values = [y / max(y_values) for y in y_values]
    print("############################## Normalized ################################")
    print("X    -values: ", x_values)
    print("--------------------------------------------------------------------------")
    print("Y-values: ", y_values)
    return x_values, y_values


def evaluate(row, w):
    prediction = 0
    right = 0
    sum_ = w[0] + w[1] * row[1] + w[2] * row[2]
    if sum_ > 0:
        prediction = 1
    else:
        prediction = 0
    if prediction == row[3]: right = 1
    return prediction, right, 1


def predict(matrix, w):
    nbr_right = 0
    predictions = []
    for row in matrix:
        prediction, right, count = evaluate(row, w)
        nbr_right += right
        predictions.append(prediction)
    return nbr_right, predictions, count


def update_step(matrix, w, alpha, predictions):
    for row, prediction in zip(matrix, predictions):
        if not prediction == row[3]:
            d = 1 if row[3] == 1 else -1
            for i in range(len(w)):
                w[i] = w[i] + alpha * d * row[i]
    return w


def train_weights(matrix, w, alpha):
    nbr_right = 0
    counter = 0
    nbr_right, predictions, count = predict(matrix, w)
    counter += count
    while nbr_right < len(matrix):
        w = update_step(matrix, w, alpha, predictions)
        nbr_right, predictions, count = predict(matrix, w)
        counter += count
    return w, counter


def plot_perceptron(X, w, alpha, *args):
    X_en = np.array([x for x in filter(lambda X: X[3] == 0, X)])
    X_fr = np.array([x for x in filter(lambda X: X[3] == 1, X)])
    x_axis = np.array(X)[:,1]
    plt.plot(x_axis, abs(w[1]) * x_axis + w[0], '-', label='Line')
    plt.scatter(X_en[:,1], X_en[:,2], marker='o', label='English, 0', color='Blue')
    plt.scatter(X_fr[:,1], X_fr[:,2], marker='o', label='French, 1', color='Red')
    language = "French"
    if args:
        if args[0][3] == 0:
            language = "English"
        args = np.array(args)
        plt.scatter(args[0][1], args[0][2], marker='o', label=f'Evaluation Sample {language} Eval: {args[1]}', color='Black')
    plt.xlabel('Letters')
    plt.ylabel('A\'s')
    plt.title(f"Perceptron classified (alpha={alpha})")
    plt.legend()
    plt.show()


