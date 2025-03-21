import numpy as np

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return -2 * (y_true - y_pred) / len(y_true)

def bce_loss(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

def bce_derivative(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred + 1e-9)) - (y_true / (y_pred + 1e-9))

def cce_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
# tambah 1e-9 -epsilon biar kalau log(0) ga masalah

def cce_derivative(y_true, y_pred):
    return -(y_true / (y_pred + 1e-9))

# kalau activation functionnya softmax
def cce_derivative_softmax(y_true, y_pred):
    return y_pred - y_true