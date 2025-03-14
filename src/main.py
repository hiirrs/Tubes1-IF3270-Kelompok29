from lossFunction import *
from activationFunction import *
from ffnn import *

layer_sizes = list(map(int, input("Number of neurons per layer (separated by space): ").split()))
activations = []
for i in range(1, len(layer_sizes)):
    act_func = input(f"Activation function for layer {i} (linear/sigmoid): ")
    if act_func == "linear":
        activations.append((linear_activation, linear_derivative))
    elif act_func == "sigmoid":
        activations.append((sigmoid_activation, sigmoid_derivative))

loss_func_input = input("Select the loss function (mse/else): ")
if loss_func_input == "mse":
    loss_func = mse_loss
    loss_derivative = mse_derivative
else:
    raise ValueError("Invalid loss function")   # tar ganti pake loss function lain

nn = FFNN(layer_sizes, activations, loss_func, loss_derivative)

# dummy data
x = np.random.rand(100, layer_sizes[0])
y = np.random.rand(100, layer_sizes[-1]) 
epochs = int(input("Number of epochs: "))
batch_size = int(input("Batch size: "))
learning_rate = float(input("Learning rate: "))

nn.train(x, y, epochs, batch_size, learning_rate)
