import numpy as np
import matplotlib.pyplot as plt
# import networkx as nx 

class FFNN:
    def __init__(self, layer_sizes, activation_funcs, loss_func, loss_derivative, initialize, init_params=None):
        self.layer_sizes = layer_sizes
        self.activation_funcs = activation_funcs
        self.loss_func = loss_func
        self.loss_derivative = loss_derivative
    
        self.weights = []
        self.biases = []

        if init_params is None:
            init_params = {}

        self.initialize_weights(initialize, init_params)

    def initialize_weights(self, method, params):
        seed = params.get("seed", None)
        if seed is not None:
            np.random.seed(int(seed))

        for i in range(len(self.layer_sizes)-1):  
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i+1]

            if method == "zero":
                w = np.zeros((in_size, out_size))
                b = np.zeros((1, out_size))

            elif method == "normal":
                mean = params.get("mean", 0)
                var = params.get("var", 1)
                w = np.random.normal(mean, np.sqrt(var), (in_size, out_size))
                b = np.random.normal(mean, np.sqrt(var), (1, out_size))

            elif method == "uniform":
                low = params.get("low", -1)
                high = params.get("high", 1)
                w = np.random.uniform(low, high, (in_size, out_size))
                b = np.random.uniform(low, high, (1, out_size))
            
            else:
                raise ValueError("Invalid initialization method")
            
            self.weights.append(w)
            self.biases.append(b)

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

    def show_structure(self):
        print(f"Feedforward Neural Network Structure Built\n")
        for i in range(len(self.weights)):
            print(f"Layer {i} → Layer {i+1}:")
            print(f"  Neurons: {self.weights[i].shape[0]} → {self.weights[i].shape[1]}")
            print(f"  Weights shape: {self.weights[i].shape}")
            print(f"  Bias shape: {self.biases[i].shape}")
    
    def plot_weight_distribution(self, layers_to_plot):
        for i in layers_to_plot:
            if i < len(self.weights):
                weights_flat = self.weights[i].flatten()
                plt.hist(weights_flat, bins=30, alpha=0.7)
                plt.title(f"Weight Distribution - Layer {i}")
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
            else:
                print(f"Layer {i} tidak valid.")
    
    def plot_gradient_distribution(self, layers_to_plot):
        if not hasattr(self, "gradients"):
            print("Gradien belum dihitung! Jalankan training dulu.")
            return

        for i in layers_to_plot:
            if i < len(self.gradients):
                grad_w_flat = self.gradients[i][0].flatten()
                plt.hist(grad_w_flat, bins=30, alpha=0.7)
                plt.title(f"Gradient Distribution - Layer {i}")
                plt.xlabel("Gradient Value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.show()
            else:
                print(f"Layer {i} tidak valid.")
