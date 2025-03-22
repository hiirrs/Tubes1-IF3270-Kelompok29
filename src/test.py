import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from ffnn_imp import FeedForwardNN  # Import our implementation

# load dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# ambil sampel (tes)
X = X[:5000]
y = y[:5000]

# Scale the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# encode
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Contoh, 0 dan 1
digit_mask = (y_train[:, 0] == 1) | (y_train[:, 1] == 1)
X_train_binary = X_train[digit_mask]
y_train_binary = y_train[digit_mask][:, :2]  # 0 dan 1 saja

digit_mask_test = (y_test[:, 0] == 1) | (y_test[:, 1] == 1)
X_test_binary = X_test[digit_mask_test]
y_test_binary = y_test[digit_mask_test][:, :2]

print(f"Binary classification data shape: {X_train_binary.shape}")
print(f"Binary classification labels shape: {y_train_binary.shape}")

# Tes 1: Binary Cross-Entropy dengan Random Normal Initialization
print("\nTest 1: Binary Cross-Entropy with Random Normal Initialization")
model_binary = FeedForwardNN(
    layer_dimensions=[784, 32, 16, 2],  # input, hidden, output
    activations=['relu', 'relu', 'sigmoid'],  # activation untuk setiap layer
    loss='binary_cross_entropy',  # fungsi loss
    weight_initializer='random_normal',  # metode init weight
    weight_init_params={'mean': 0.0, 'variance': 0.01, 'seed': 42}  # param weight init
)

# latih
history_binary = model_binary.train(
    X_train_binary, y_train_binary, 
    X_val=X_test_binary, y_val=y_test_binary,  
    batch_size=32, 
    learning_rate=0.01, 
    epochs=10,
    verbose=1
)

# Tes 2: Categorical Cross-Entropy dengan Softmax (Multi-class classification)
print("\nTest 2: Categorical Cross-Entropy with Softmax (Multi-class)")
model_softmax = FeedForwardNN(
    layer_dimensions=[784, 64, 32, 10],  # input, hidden, output (10 kelas)
    activations=['relu', 'relu', 'softmax'],  # softmax untuk output layer
    loss='categorical_cross_entropy',  
    weight_initializer='random_normal', 
    weight_init_params={'mean': 0.0, 'variance': 0.01, 'seed': 42}  
)

# latih
history_softmax = model_softmax.train(
    X_train, y_train, 
    X_val=X_test, y_val=y_test,  
    batch_size=32, 
    learning_rate=0.01, 
    epochs=10,
    verbose=1
)

# bandingin
print("\nComparison with sklearn's MLP")
sklearn_mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='sgd',
    alpha=0.0,
    batch_size=32,
    learning_rate_init=0.01,
    max_iter=10,
    random_state=42
)

sklearn_mlp.fit(X_train, np.argmax(y_train, axis=1)) 
sklearn_score = sklearn_mlp.score(X_test, np.argmax(y_test, axis=1))
print(f"sklearn MLPClassifier accuracy: {sklearn_score:.4f}")

# prediksi
softmax_preds = model_softmax.predict(X_test)
softmax_accuracy = np.mean(np.argmax(softmax_preds, axis=1) == np.argmax(y_test, axis=1))
print(f"Our FFNN with softmax accuracy: {softmax_accuracy:.4f}")

# visualisasi
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_binary['train_loss'], label='Train Loss')
plt.plot(history_binary['val_loss'], label='Validation Loss')
plt.title('Binary Cross-Entropy Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_softmax['train_loss'], label='Train Loss')
plt.plot(history_softmax['val_loss'], label='Validation Loss')
plt.title('Categorical Cross-Entropy with Softmax')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

model_softmax.plot_weight_distribution()
plt.savefig('weight_distribution.png')
plt.show()

model_softmax.plot_gradient_distribution()
plt.savefig('gradient_distribution.png')
plt.show()

# model_softmax.visualize_model()
# plt.savefig('model_visualization.png')
# plt.show()

# save
model_softmax.save('ffnn_model.pkl')
print("Model saved to 'ffnn_model.pkl'")