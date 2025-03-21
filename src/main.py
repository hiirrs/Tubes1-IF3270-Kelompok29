from lossFunction import *
from activationFunction import *
from ffnn import *

layer_sizes = list(map(int, input("Number of neurons per layer (separated by space): ").split()))
activations = []
# for i in range(1, len(layer_sizes)):
#     act_func = input(f"Activation function for layer {i} (linear/sigmoid): ")
#     if act_func == "linear":
#         activations.append((linear_activation, linear_derivative))
#     elif act_func == "sigmoid":
#         activations.append((sigmoid_activation, sigmoid_derivative))
activations_map = {
    "linear": (linear_activation, linear_derivative),
    "sigmoid": (sigmoid_activation, sigmoid_derivative),
    "relu": (ReLu_activation, ReLu_derivative),
    "tanh": (tanh_activation, tanh_derivative),
    "softmax": (softmax_activation, softmax_derivative)
}

for i in range(1, len(layer_sizes)):
    act_func = input(f"Activation function for layer {i} (linear/sigmoid/relu/tanh/softmax): ").lower()
    if act_func in activations_map:
        activations.append(activations_map[act_func])
    else:
        raise ValueError("Invalid activation function")


loss_func_input = input("Select the loss function (mse/cce/bce/else-comingsoon): ")
if loss_func_input == "mse":
    loss_func = mse_loss
    loss_derivative = mse_derivative
elif loss_func_input == "cce":
    loss_func = cce_loss   
    if activations[-1][0] == softmax_activation:
        loss_derivative = cce_derivative_softmax
    else: 
        loss_derivative = cce_derivative       
elif loss_func_input == "bce":
    loss_func = bce_loss
    loss_derivative = bce_derivative
else:
    raise ValueError("Invalid loss function")  

initialization = input("Select the initialization method (zero/normal/uniform): ")
init_params = {}

if initialization == "normal":
    init_params["mean"] = float(input("Mean: "))    
    init_params["var"] = float(input("Variance: "))
    init_params["seed"] = int(input("Seed (integer): "))
elif initialization == "uniform":   
    init_params["low"] = float(input("Lowwer Bound: "))    
    init_params["high"] = float(input("Upper Bound: "))    
    init_params["seed"] = float(input("Seed (integer): "))    
else:
    pass

nn = FFNN(layer_sizes, activations, loss_func, loss_derivative, initialization, init_params)

# dummy data
# x = np.random.rand(100, layer_sizes[0])
# y = np.random.rand(100, layer_sizes[-1]) 
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([
    [1],
    [0.05],
    [0.1]
])
epochs = int(input("Number of epochs: "))
batch_size = int(input("Batch size: "))
learning_rate = float(input("Learning rate: "))

nn.train(x, y, epochs, batch_size, learning_rate)
nn.show_structure()
nn.plot_weight_distribution([0, 1])  # untuk layer 0 dan 1
nn.plot_gradient_distribution([0, 1])
