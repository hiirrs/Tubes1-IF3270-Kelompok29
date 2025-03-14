import numpy as np

class FFNN:
    def __init__(self, layer_sizes, activation_funcs, loss_func, loss_derivative):
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs
        self.loss_func = loss_func
        self.loss_derivative = loss_derivative
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.biases = [np.random.randn(1, layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]

    def forward(self, x):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(activations[-1], w) + b
            a = self.activation_funcs[i][0](z)
            activations.append(a)
        return activations

    def backward(self, activations, y_true):
        self.gradients = []
        delta = self.loss_derivative(y_true, activations[-1]) * self.activation_funcs[-1][1](activations[-1])
        for i in reversed(range(len(self.weights))):
            delta_w = np.dot(activations[i].T, delta)
            delta_b = np.sum(delta, axis=0)
            self.gradients.insert(0, (delta_w, delta_b))
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_funcs[i-1][1](activations[i])

    def update_weights(self, lr):
        for i, (dw, db) in enumerate(self.gradients):
            self.weights[i] -= lr * dw
            self.biases[i] -= lr * db

    def train(self, x, y, epochs, batch_size, lr):
        for epoch in range(epochs):
            for start in range(0, len(x), batch_size):
                end = start + batch_size
                activations = self.forward(x[start:end])
                self.backward(activations, y[start:end])
                self.update_weights(lr)
            loss = self.loss_func(y, self.forward(x)[-1])
            print(f"Epoch {epoch+1}, Loss: {loss}")
