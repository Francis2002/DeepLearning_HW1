import numpy as np
import utils

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layers, output_size):
        # Initialize weights and biases for each layer
        self.layers = []
        layer_sizes = [input_size] + hidden_layers + [output_size]

        for i in range(len(layer_sizes) - 1):
            weights = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            biases = np.zeros((1, layer_sizes[i+1]))
            self.layers.append((weights, biases))

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        activations = X

        for i, (weights, biases) in enumerate(self.layers):
            z = np.dot(activations, weights) + biases
            if i == len(self.layers) - 1:
                activations = self.softmax(z)
            else:
                activations = self.relu(z)

        return activations

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihoods = -np.log(y_pred[range(m), y_true])
        loss = np.sum(log_likelihoods) / m
        return loss

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Compute loss
            loss = self.compute_loss(y_pred, y)

            # Backward pass
            gradients = self.backward(X, y, y_pred)

            # Update weights and biases
            self.update_params(gradients, learning_rate)

            if epoch % 10 == 0:
                print(f'Epoch {epoch}, loss: {loss}')

    def backward(self, X, y, y_pred):
        # Backpropagation
        m = X.shape[0]
        gradients = []

        # Gradient of the loss with respect to the output of the last layer (softmax)
        dZ = y_pred.copy()
        dZ[range(m), y] -= 1
        dZ /= m

        for i in reversed(range(len(self.layers))):
            weights, _ = self.layers[i]
            if i != 0:
                # If not the first hidden layer
                prev_activations = self.relu(np.dot(X, self.layers[i-1][0]) + self.layers[i-1][1])
            else:
                # If the first hidden layer, previous activations are the input features
                prev_activations = X

            # Calculate gradients
            dW = np.dot(prev_activations.T, dZ)
            dB = np.sum(dZ, axis=0, keepdims=True)

            # For the next layer
            if i != 0:
                dA = np.dot(dZ, weights.T)
                dZ = dA * (prev_activations > 0)  # Derivative of ReLU

            gradients.insert(0, (dW, dB))

        return gradients

    def update_params(self, gradients, learning_rate):
        for i, (dW, dB) in enumerate(gradients):
            weights, biases = self.layers[i]
            weights -= learning_rate * dW
            biases -= learning_rate * dB
            self.layers[i] = (weights, biases)


# Example usage
input_size = 784  # for MNIST dataset
hidden_layers = [200]  # one hidden layers with 200 neurons
output_size = 4  # for MNIST dataset, 10 classes

mlp = MultiLayerPerceptron(input_size, hidden_layers, output_size)

# For training, you would need to provide the input data X and labels y
add_bias = False
data = utils.load_oct_data(bias=add_bias)
train_X, train_y = data["train"]
dev_X, dev_y = data["dev"]
test_X, test_y = data["test"]
n_classes = np.unique(train_y).size
n_feats = train_X.shape[1]

mlp.train(train_X, train_y, epochs=1000, learning_rate=0.01)
