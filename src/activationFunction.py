import numpy as np

# linear
def linear_activation(x):
    return x

def linear_derivative(x):
    return 1

# ReLu
def ReLu_activation(x):
    return np.maximum(0, x)

def ReLu_derivative(x):
    return np.where(x > 0, 1, 0)

# hyperbolic
def tanh_activation(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

# sigmoid
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid_activation(x) * (1 - sigmoid_activation(x))

# softmax
def softmax_activation(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # stabilisasi numerik
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# softmax derivative (biasanya digabung dengan loss dalam backprop)
def softmax_derivative(x):
    # digunakan hanya jika loss bukan CCE
    s = softmax_activation(x)
    return s * (1 - s)  # simplifikasi, biasanya ga dipakai eksplisit
