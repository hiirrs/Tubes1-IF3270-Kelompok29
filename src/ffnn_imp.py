import numpy as np
import matplotlib.pyplot as plt
from typing import List
import networkx as nx
import pickle

class Activation:
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def linear_derivative(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Hindari overflow
    
    @staticmethod
    def sigmoid_derivative(x):
        sigmoid_x = Activation.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        # Di backward propagation
        pass


class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    
    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / y_true.shape[0]


class WeightInitializer:
    @staticmethod
    def zero_initialization(shape):
        return np.zeros(shape)
    
    @staticmethod
    def random_uniform(shape, lower_bound=-0.5, upper_bound=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(lower_bound, upper_bound, shape)
    
    @staticmethod
    def random_normal(shape, mean=0.0, variance=0.1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(mean, np.sqrt(variance), shape)


class Layer:
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        activation: str = 'linear',
        weight_initializer: str = 'random_normal',
        weight_init_params: dict = None
    ):
        self.input_size = input_size
        self.output_size = output_size
        
        self.activation_name = activation
        if activation == 'linear':
            self.activation = Activation.linear
            self.activation_derivative = Activation.linear_derivative
        elif activation == 'relu':
            self.activation = Activation.relu
            self.activation_derivative = Activation.relu_derivative
        elif activation == 'sigmoid':
            self.activation = Activation.sigmoid
            self.activation_derivative = Activation.sigmoid_derivative
        elif activation == 'tanh':
            self.activation = Activation.tanh
            self.activation_derivative = Activation.tanh_derivative
        elif activation == 'softmax':
            self.activation = Activation.softmax
            self.activation_derivative = None  #Backward pass
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # init weights and biases
        if weight_init_params is None:
            weight_init_params = {}
            
        shape_weights = (input_size, output_size)
        shape_bias = (1, output_size)
        
        if weight_initializer == 'zero':
            self.weights = WeightInitializer.zero_initialization(shape_weights)
            self.bias = WeightInitializer.zero_initialization(shape_bias)
        elif weight_initializer == 'random_uniform':
            lower_bound = weight_init_params.get('lower_bound', -0.5)
            upper_bound = weight_init_params.get('upper_bound', 0.5)
            seed = weight_init_params.get('seed', None)
            self.weights = WeightInitializer.random_uniform(shape_weights, lower_bound, upper_bound, seed)
            self.bias = WeightInitializer.random_uniform(shape_bias, lower_bound, upper_bound, seed)
        elif weight_initializer == 'random_normal':
            mean = weight_init_params.get('mean', 0.0)
            variance = weight_init_params.get('variance', 0.1)
            seed = weight_init_params.get('seed', None)
            self.weights = WeightInitializer.random_normal(shape_weights, mean, variance, seed)
            self.bias = WeightInitializer.random_normal(shape_bias, mean, variance, seed)
        else:
            raise ValueError(f"Unsupported weight initializer: {weight_initializer}")
        
        # init gradients
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        
        self.input = None
        self.output = None
        self.linear_output = None
    
    def forward(self, x):
        self.input = x
        self.linear_output = np.dot(x, self.weights) + self.bias
        self.output = self.activation(self.linear_output)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        if self.activation_name == 'softmax':
            batch_size = output_gradient.shape[0]
            linear_gradient = output_gradient
        else:
            activation_gradient = self.activation_derivative(self.linear_output)
            linear_gradient = output_gradient * activation_gradient
        
        self.weights_gradient = np.dot(self.input.T, linear_gradient)
        self.bias_gradient = np.sum(linear_gradient, axis=0, keepdims=True)
        
        input_gradient = np.dot(linear_gradient, self.weights.T)
        
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
        
        return input_gradient


class FeedForwardNN:
    def __init__(self, layer_dimensions: List[int], activations: List[str], loss: str = 'mse', 
                 weight_initializer: str = 'random_normal', weight_init_params: dict = None):

        if len(layer_dimensions) < 2:
            raise ValueError("Network must have at least input and output layers")
        
        if len(activations) != len(layer_dimensions) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")
        
        # loss function
        self.loss_name = loss
        if loss == 'mse':
            self.loss_function = Loss.mse
            self.loss_derivative = Loss.mse_derivative
        elif loss == 'binary_cross_entropy':
            self.loss_function = Loss.binary_cross_entropy
            self.loss_derivative = Loss.binary_cross_entropy_derivative
        elif loss == 'categorical_cross_entropy':
            self.loss_function = Loss.categorical_cross_entropy
            self.loss_derivative = Loss.categorical_cross_entropy_derivative
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        
        # init layers
        self.layers = []
        for i in range(len(layer_dimensions) - 1):
            self.layers.append(
                Layer(
                    input_size=layer_dimensions[i],
                    output_size=layer_dimensions[i+1],
                    activation=activations[i],
                    weight_initializer=weight_initializer,
                    weight_init_params=weight_init_params
                )
            )
        
        self.layer_dimensions = layer_dimensions
        self.activations = activations
    
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, y_true, y_pred, learning_rate):
        if self.activations[-1] == 'softmax' and self.loss_name == 'categorical_cross_entropy':
            gradient = y_pred - y_true
        else:
            gradient = self.loss_derivative(y_true, y_pred)
        
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              batch_size=32, learning_rate=0.01, epochs=100, verbose=1):
    
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        validate = X_val is not None and y_val is not None
        if validate:
            X_val = np.array(X_val)
            y_val = np.array(y_val)
        
        n_samples = X_train.shape[0]
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            total_loss = 0
            batch_count = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                y_pred = self.forward(X_batch)
                
                batch_loss = self.loss_function(y_batch, y_pred)
                total_loss += batch_loss
                batch_count += 1
                
                self.backward(y_batch, y_pred, learning_rate)
            
            avg_train_loss = total_loss / batch_count
            history['train_loss'].append(avg_train_loss)
            
            if validate:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function(y_val, y_val_pred)
                history['val_loss'].append(val_loss)
            
            if verbose == 1:
                if validate:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return history
    
    def predict(self, X):
        return self.forward(X)
    
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        loss = self.loss_function(y_true, y_pred)
        return loss
    
    def visualize_model(self, figsize=(10, 8)):
        G = nx.DiGraph()
        
        layer_nodes = []
        for i, dim in enumerate(self.layer_dimensions):
            layer_nodes.append([])
            for j in range(dim):
                node_id = f"L{i}_N{j}"
                G.add_node(node_id, layer=i, neuron=j)
                layer_nodes[i].append(node_id)
        
        for layer_idx, layer in enumerate(self.layers):
            for i in range(self.layer_dimensions[layer_idx]):
                for j in range(self.layer_dimensions[layer_idx + 1]):
                    source = layer_nodes[layer_idx][i]
                    target = layer_nodes[layer_idx + 1][j]
                    weight = layer.weights[i, j]
                    G.add_edge(source, target, weight=weight, gradient=layer.weights_gradient[i, j])
        
        pos = {}
        for i, layer_list in enumerate(layer_nodes):
            for j, node in enumerate(layer_list):
                y_pos = j - len(layer_list) / 2
                pos[node] = (i, y_pos)
        
        plt.figure(figsize=figsize)
        
        for i, layer_list in enumerate(layer_nodes):
            name = "Input" if i == 0 else "Output" if i == len(layer_nodes) - 1 else f"Hidden {i}"
            nx.draw_networkx_nodes(G, pos, nodelist=layer_list, node_color=f"C{i}", 
                                  node_size=500, label=name)
        
        for u, v, data in G.edges(data=True):
            weight = data['weight']
            gradient = data['gradient']
            if weight < 0:
                color = (1, 0, 0, min(1, abs(weight)))  
            else:
                color = (0, 0.7, 0, min(1, abs(weight)))  
            
            width = 1 + 5 * abs(weight) / (max(0.1, abs(weight)))
            
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[color], width=width)
        
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Neural Network Structure with Weights and Gradients")
        plt.legend()
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def plot_weight_distribution(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.layers)))
        
        n_layers = len(layers_to_plot)
        fig, axs = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))
        
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx >= len(self.layers):
                continue
                
            layer = self.layers[layer_idx]
            weights = layer.weights.flatten()
            
            axs[i].hist(weights, bins=50, color='blue', alpha=0.7)
            axs[i].set_title(f"Layer {layer_idx+1} Weight Distribution")
            axs[i].set_xlabel("Weight Value")
            axs[i].set_ylabel("Frequency")
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def plot_gradient_distribution(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = list(range(len(self.layers)))
        
        n_layers = len(layers_to_plot)
        fig, axs = plt.subplots(1, n_layers, figsize=(n_layers * 5, 5))
        
        if n_layers == 1:
            axs = [axs]
        
        for i, layer_idx in enumerate(layers_to_plot):
            if layer_idx >= len(self.layers):
                continue
                
            layer = self.layers[layer_idx]
            gradients = layer.weights_gradient.flatten()
            
            axs[i].hist(gradients, bins=50, color='red', alpha=0.7)
            axs[i].set_title(f"Layer {layer_idx+1} Gradient Distribution")
            axs[i].set_xlabel("Gradient Value")
            axs[i].set_ylabel("Frequency")
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)