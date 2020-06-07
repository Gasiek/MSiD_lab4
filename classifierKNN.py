import mnist_reader
import numpy as np


def load_data():
    X_train, y_train = mnist_reader.load_mnist('fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('fashion', kind='t10k')
    data = {'Xval': X_test,
            'Xtrain': X_train,
            'yval': y_test,
            'ytrain': y_train}
    return data

def manhattan_distance(X, X_train):
    X = X.astype(int)
    X_train = X_train.astype(int)
    Dist = np.zeros((len(X), len(X_train)))
    for i in range(0, len(X)):
        Dist[i] = np.sum(abs(X_train - X[i]), axis=1)
        print(i)
    return Dist


def sort_train_labels_knn(Dist, y):
    Dist = Dist.astype(int)
    y = y.astype(int)
    index_array = np.argsort(Dist, kind='mergesort', axis=1)
    return y[index_array]


def p_y_x_knn(y, k):
    y = y.astype(int)
    theme_classes = np.sort(np.unique(y[0]))
    result = []
    for row in y:
        result.append([np.sum(row[0:k] == x) for x in theme_classes])
    result = np.array(result) / k
    return result


def classification_error(p_y_x, y_true):
    predict_labels = np.argmax(p_y_x, axis=1)
    result = np.sum(y_true != predict_labels) / y_true.size
    return result


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    Xval = Xval.astype(int)
    Xtrain = Xtrain.astype(int)
    yval = yval.astype(int)
    ytrain = ytrain.astype(int)

    dist_matrix = manhattan_distance(Xval, Xtrain)
    sort_dist_matrix = sort_train_labels_knn(dist_matrix, ytrain)
    errors = [classification_error(p_y_x_knn(sort_dist_matrix, k), yval) for k in k_values]
    best_error = np.amin(errors)
    best_k = k_values[np.argmin(errors)]
    return best_error, best_k, errors

def run_training():

    data = load_data()
    # print(data['Xtrain'].shape)
    # KNN model selection
    k_values = range(1, 10)
    print('\n------------- Selekcja liczby sasiadow dla modelu dla KNN -------------')
    print('-------------------- Wartosci k: 1, 2, 3, 4, 5, 6, 7, 8, 9 -----------------------')

    error_best, best_k, errors = model_selection_knn(data['Xval'],
                                                     data['Xtrain'],
                                                     data['yval'],
                                                     data['ytrain'],
                                                     k_values)
    print('Najlepsze k: {num1} i najlepszy blad: {num2:.4f}'.format(num1=best_k, num2=error_best))


run_training()
