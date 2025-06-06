import numpy as np
import matplotlib.pyplot as plt
from typing import List
from tqdm.auto import trange
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
        # x = np.clip(x, -500, 500) 
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(x):
        # Di backward propagation
        pass

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))


class Loss:
    @staticmethod
    def mse(y_true, y_pred, model=None, l1_lambda=0, l2_lambda=0):
        mse_loss = np.mean(np.square(y_true - y_pred))
        
        if model is not None:
            l1_reg, l2_reg = Loss.calculate_regularization(model, l1_lambda, l2_lambda)
            return mse_loss + l1_reg + l2_reg
        
        return mse_loss
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_cross_entropy(y_true, y_pred, model=None, l1_lambda=0, l2_lambda=0):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        bce_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if model is not None:
            l1_reg, l2_reg = Loss.calculate_regularization(model, l1_lambda, l2_lambda)
            return bce_loss + l1_reg + l2_reg
        
        return bce_loss
    
    @staticmethod
    def binary_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(y_true / y_pred - (1 - y_true) / (1 - y_pred)) / y_true.shape[0]
    
    @staticmethod
    def categorical_cross_entropy(y_true, y_pred, model=None, l1_lambda=0, l2_lambda=0):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cce_loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        
        if model is not None:
            l1_reg, l2_reg = Loss.calculate_regularization(model, l1_lambda, l2_lambda)
            return cce_loss + l1_reg + l2_reg
        
        return cce_loss
    
    @staticmethod
    def categorical_cross_entropy_derivative(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred / y_true.shape[0]
    
    @staticmethod
    def calculate_regularization(model, l1_lambda, l2_lambda):
        l1_reg = 0
        l2_reg = 0
        
        if l1_lambda > 0 or l2_lambda > 0:
            for layer in model.layers:
                if l1_lambda > 0:
                    l1_reg += l1_lambda * np.sum(np.abs(layer.weights))
                
                if l2_lambda > 0:
                    l2_reg += l2_lambda * np.sum(np.square(layer.weights)) / 2
        
        return l1_reg, l2_reg


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
    
    @staticmethod
    def xavier(shape, fan_in, fan_out, seed=None):
        if seed is not None:
            np.random.seed(seed)
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, shape)

    @staticmethod
    def he(shape, fan_in, seed=None):
        if seed is not None:
            np.random.seed(seed)
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(0.0, stddev, shape)



class Layer:
    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        activation: str = 'linear',
        weight_initializer: str = 'random_normal',
        weight_init_params: dict = None,
        normalization: str = None
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
        elif activation == 'leaky_relu':
            alpha = weight_init_params.get('alpha', 0.01) if weight_init_params else 0.01
            self.activation = lambda x: Activation.leaky_relu(x, alpha)
            self.activation_derivative = lambda x: Activation.leaky_relu_derivative(x, alpha)
        elif activation == 'elu':
            alpha = weight_init_params.get('alpha', 1.0) if weight_init_params else 1.0
            self.activation = lambda x: Activation.elu(x, alpha)
            self.activation_derivative = lambda x: Activation.elu_derivative(x, alpha)
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
        elif weight_initializer == 'xavier':
            seed = weight_init_params.get('seed', None)
            self.weights = WeightInitializer.xavier(shape_weights, self.input_size, self.output_size, seed)
            self.bias = WeightInitializer.xavier(shape_bias, self.input_size, self.output_size, seed)
        elif weight_initializer == 'he':
            seed = weight_init_params.get('seed', None)
            self.weights = WeightInitializer.he(shape_weights, self.input_size, seed)
            self.bias = WeightInitializer.he(shape_bias, self.input_size, seed)
        elif weight_initializer == 'auto':
            if self.activation_name in ['relu', 'leaky_relu']:
                weight_initializer = 'he'
            elif self.activation_name in ['sigmoid', 'tanh']:
                weight_initializer = 'xavier'
            else:
                weight_initializer = 'random_normal'
        else:
            raise ValueError(f"Unsupported weight initializer: {weight_initializer}")
        
        # init gradients
        self.weights_gradient = np.zeros_like(self.weights)
        self.bias_gradient = np.zeros_like(self.bias)
        
        self.input = None
        self.output = None
        self.linear_output = None

        self.normalization = normalization
        if normalization == 'rmsnorm':
            self.norm_layer = RMSNorm(output_size)
        else:
            self.norm_layer = None
    
    def forward(self, x):
        self.input = x
        self.linear_output = np.dot(x, self.weights) + self.bias

        if self.norm_layer is not None:
            self.linear_output = self.norm_layer.forward(self.linear_output)

        self.output = self.activation(self.linear_output)
        return self.output
    
    def backward(self, output_gradient, learning_rate, l1_lambda=0, l2_lambda=0):
        if self.activation_name == 'softmax':
            batch_size = output_gradient.shape[0]
            linear_gradient = output_gradient
        else:
            activation_gradient = self.activation_derivative(self.linear_output)
            linear_gradient = output_gradient * activation_gradient
        
        if self.norm_layer is not None:
            linear_gradient = self.norm_layer.backward(linear_gradient, learning_rate)

        self.weights_gradient = np.dot(self.input.T, linear_gradient)
        self.bias_gradient = np.sum(linear_gradient, axis=0, keepdims=True)
        
        if l1_lambda > 0:
            l1_grad = l1_lambda * np.sign(self.weights)
            self.weights_gradient += l1_grad
        
        if l2_lambda > 0:
            l2_grad = l2_lambda * self.weights
            self.weights_gradient += l2_grad
        
        input_gradient = np.dot(linear_gradient, self.weights.T)
        
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient
        
        return input_gradient
    


class FeedForwardNN:
    def __init__(
            self, 
            layer_dimensions: List[int], 
            activations: List[str], 
            loss: str = 'mse', 
            weight_initializer: str = 'random_normal', 
            weight_init_params: dict = None,
            normalization: List[str] = None,
            l1_lambda: float = 0,  
            l2_lambda: float = 0   
        ):
        
        if len(layer_dimensions) < 2:
            raise ValueError("Network must have at least input and output layers")
        if len(activations) != len(layer_dimensions) - 1:
            raise ValueError("Number of activation functions must match number of layers - 1")

        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
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
        
        if normalization is None:
            normalization = [None] * (len(layer_dimensions) - 1)
        
        if len(normalization) != len(layer_dimensions) - 1:
            raise ValueError("Number of normalization methods must match number of layers - 1")

        self.layers = [Layer(input_size=layer_dimensions[i], output_size=layer_dimensions[i+1],
                            activation=activations[i], weight_initializer=weight_initializer, 
                            weight_init_params=weight_init_params) for i in range(len(layer_dimensions) - 1)]

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
            gradient = layer.backward(gradient, learning_rate, self.l1_lambda, self.l2_lambda)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        loss = self.loss_function(y_true, y_pred, model=self, l1_lambda=self.l1_lambda, l2_lambda=self.l2_lambda)
        return loss
    
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
                
                batch_loss = self.loss_function(
                    y_batch, y_pred, 
                    model=self, 
                    l1_lambda=self.l1_lambda, 
                    l2_lambda=self.l2_lambda
                )
                total_loss += batch_loss
                batch_count += 1
                
                self.backward(y_batch, y_pred, learning_rate)
            
            avg_train_loss = total_loss / batch_count
            history['train_loss'].append(avg_train_loss)
            
            if validate:
                y_val_pred = self.forward(X_val)
                val_loss = self.loss_function(
                    y_val, y_val_pred, 
                    model=self, 
                    l1_lambda=self.l1_lambda, 
                    l2_lambda=self.l2_lambda
                )
                history['val_loss'].append(val_loss)
            
            if verbose == 1:
                if validate:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return history
    
    def predict(self, X):
        return self.forward(X)
    
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
    
    # Modified plot_weight_distribution method to handle NaN values
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
            
            if np.isnan(weights).any() or np.isinf(weights).any():
                print(f"Warning: Layer {layer_idx+1} contains NaN or Inf values in weights. Filtering for visualization.")
                # remove nan dan inf values for visualization
                valid_weights = weights[~np.isnan(weights) & ~np.isinf(weights)]
                
                if len(valid_weights) > 0:
                    axs[i].hist(valid_weights, bins=50, color='blue', alpha=0.7)
                    axs[i].set_title(f"Layer {layer_idx+1} Weight Distribution\n(filtered {np.isnan(weights).sum()} NaN values)")
                else:
                    axs[i].text(0.5, 0.5, "All weights are NaN/Inf", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[i].transAxes)
                    axs[i].set_title(f"Layer {layer_idx+1} Weight Distribution\n(all values are NaN/Inf)")
            else:
                axs[i].hist(weights, bins=50, color='blue', alpha=0.7)
                axs[i].set_title(f"Layer {layer_idx+1} Weight Distribution")
                
            axs[i].set_xlabel("Weight Value")
            axs[i].set_ylabel("Frequency")
            axs[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt

    # Modified plot_gradient_distribution method to handle NaN values
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
            
            # Handle NaN values in gradients
            if np.isnan(gradients).any() or np.isinf(gradients).any():
                print(f"Warning: Layer {layer_idx+1} contains NaN or Inf values in gradients. Filtering for visualization.")
                # Filter out NaN and Inf values for visualization
                valid_gradients = gradients[~np.isnan(gradients) & ~np.isinf(gradients)]
                
                if len(valid_gradients) > 0:
                    axs[i].hist(valid_gradients, bins=50, color='red', alpha=0.7)
                    axs[i].set_title(f"Layer {layer_idx+1} Gradient Distribution\n(filtered {np.isnan(gradients).sum()} NaN values)")
                else:
                    axs[i].text(0.5, 0.5, "All gradients are NaN/Inf", 
                            horizontalalignment='center', verticalalignment='center',
                            transform=axs[i].transAxes)
                    axs[i].set_title(f"Layer {layer_idx+1} Gradient Distribution\n(all values are NaN/Inf)")
            else:
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
        
    def visualize_nn_graph(self, figsize=(12, 6)):     
        G = nx.DiGraph()
        layer_names = []
        
        for layer_idx, layer in enumerate(self.layers):
            layer_name = f'Layer {layer_idx}'
            layer_names.append(layer_name)
            
            for node_idx in range(layer.output_size):
                node_id = f"{layer_name} Neuron {node_idx}"
                G.add_node(node_id)
                G.nodes[node_id]['value'] = layer.bias[0][node_idx]  
                G.nodes[node_id]['layer'] = layer_idx
                
                if layer_idx > 0:
                    prev_layer = self.layers[layer_idx - 1]
                    for prev_node_idx in range(prev_layer.output_size):
                        prev_node_id = f"Layer {layer_idx - 1} Neuron {prev_node_idx}"
                        weight = prev_layer.weights[prev_node_idx, node_idx]
                        G.add_edge(prev_node_id, node_id, weight=weight)
        
        pos = nx.multipartite_layout(G, subset_key="layer")
        labels = {}
        
        for node in G.nodes(data=True):
            node_id = node[0]
            value = node[1]['value'] if 'value' in node[1] else ''
            labels[node_id] = value  

        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightgray', arrows=True)
        
        edge_labels = {(u, v): f"{data['weight']:.4f}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        
        plt.title("Neural Network Structure")
        plt.show()
        

class RMSNorm:
    def __init__(self, num_features, epsilon=1e-8, learnable_scale=True):
        self.num_features = num_features
        self.epsilon = epsilon
        
        # learnable scaling parameter
        if learnable_scale:
            self.scale = np.ones((1, num_features))
        else:
            self.scale = None
        
        # store inputs buat backward 
        self.inputs = None
        self.normalized_inputs = None
        
    def forward(self, x):
        self.inputs = x
        
        rms = np.sqrt(np.mean(x**2, axis=1, keepdims=True) + self.epsilon)
        
        # normalize
        self.normalized_inputs = x / rms
        
        # apply learnable scale
        if self.scale is not None:
            return self.normalized_inputs * self.scale
        
        return self.normalized_inputs
    
    def backward(self, output_gradient, learning_rate=0.01):
        rms = np.sqrt(np.mean(self.inputs**2, axis=1, keepdims=True) + self.epsilon)
        
        # itung gradient 
        dx_normalized = output_gradient
        
        # apply scale gradient
        if self.scale is not None:
            scale_gradient = np.sum(self.normalized_inputs * output_gradient, axis=0, keepdims=True)
            self.scale -= learning_rate * scale_gradient
            dx_normalized *= self.scale
        
        # itung input gradient
        dx = (
            (rms * dx_normalized - self.inputs * (np.mean(self.inputs * dx_normalized, axis=1, keepdims=True) / (rms**2))) / rms
        )
        
        return dx
