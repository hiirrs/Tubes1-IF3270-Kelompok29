<h1 align="center"> Tugas Besar 1 IF3270 Pembelajaran Mesin </h1>
<h1 align="center">  Feedforward Neural Network </h1>

## Table of Contents
1. [General Information](#general-information)
2. [Contributors](#contributors)
3. [Features](#features)
4. [Requirements Program](#required_program)
5. [How to Run The Program](#how-to-run-the-program)
6. [Project Status](#project-status)
7. [Project Structure](#project-structure)

## General Information
This project implements a Feedforward Neural Network (FFNN) from scratch using Python and Numpy. The FFNN model involves various components such as forward propagation, backward propagation, and weight updates using gradient descent. We also experiment with hyperparameters like learning rate, width, and depth of the network, comparing different activation functions (e.g., Linear, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU), and exploring weight initialization methods. The performance of the model is analyzed and compared with the sklearn MLPClassifier.


## Contributors
### **Kelompok 29 - K1**
|   NIM    |                  Nama                  | Pembagian Tugas |
| :------: |  ------------------------------------  |  -------------  |
| 13522053 |       Erdianti Wiga Putri Andini       | Linear, Sigmoid, Random Distribusi Uniform, MSE, Forward, Backward, Visualisasi Network, Normalisasi RMSNorm, Laporan |
| 13522063 |         Shazya Audrea Taufik           | Binary Cross Entropy, Random Distribusi Normal, Softmax, Forward, Backward, Kode pengujian, Visualisasi gradien dan bobot, Laporan |
| 13522085 |          Zahira Dina Amalia            | Categorical cross entropy, Zero weight initialization, ReLU, Hyperbolic, He, Xavier, Leaky ReLU, eLU, Forward, Backward, Laporan |


## Features
Features that used in this program are:
| NO  | Feature                     | Description                                                           |
|:---:|-----------------------------|-----------------------------------------------------------------------|
| 1   | Feedforward Neural Network  | Implemented from scratch with forward and backward propagation        |
| 2   | Activation Functions        | Linear, Sigmoid, Tanh, ReLU, Leaky ReLU, and ELU                      | 
| 3   | Loss Functions              | Supports MSE, Binary Cross-Entropy, and Categorical Cross-Entropy     | 
| 4   | Weight Initialization       | Random, Xavier, He, Zero Initialization                               |
| 5   | Hyperparameter Tuning       | Experiments on learning rate, depth, width, and batch size            |


## Requirements Program
|   NO   |  Required Program    |                           Reference Link                                                               |
| :----: | ---------------------|--------------------------------------------------------------------------------------------------------|
|   1    | Python 3.x           | [Python](https://www.python.org/downloads/)                                                            |                            
|   2    | Numpy                | [Numpy](https://numpy.org/)                                                                            |
|   3    | Matplotlib           | [Matplotlib](https://matplotlib.org/)                                                                  |
|   4    | sklearn              | [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html) |


## How to Run The Program
### Clone Repository
1. Open terminal.
2. Clone the repository by typing `git clone https://github.com/hiirrs/Tubes1-IF3270-Kelompok29.git` in the terminal.

### Run the Program:
1. If you don't have the requirements, install them by typing:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
2. To run the Notebook for interactive experimentation:
    ```bash
    jupyter notebook FFNN.ipynb
    ```
3. Alternatively you can run the Notebook by clicking the `Run All` button


## Project Status
This project has been completed and can be executed.


## Project Structure
```bash

│
├── README.md
│
├── doc/                        # Document files
│   └── Tubes1_IF3270_Kelompok29.pdf
|
└── src
    ├── ffnn_imp.py             # Functions
    └── FFNN.ipynb              # Run and analysis


