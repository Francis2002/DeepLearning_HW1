#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        
        #make prediction
        y_hat = self.predict(x_i)

        #update weights if prediction is wrong according to linear regression
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        
        #make prediction
        scores = np.dot(self.W, x_i.T)

        #one hot encoding
        y_true = np.zeros(self.W.shape[0])
        y_true[y_i] = 1

        #Softmax activation function
        probs = np.exp(scores) / np.sum(np.exp(scores))

        #update weights
        self.W -= learning_rate * np.outer(probs - y_true, x_i)



class MLP(object):
    # Q1.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # Intitialize weights and biases with N(0.1, 0.1^2) (N(miu, sigma^2))
        self.W1 = np.random.normal(0.1, 0.1**2, (hidden_size, n_features))
        self.W2 = np.random.normal(0.1, 0.1**2, (n_classes, hidden_size))
        self.b1 = np.random.normal(0.1, 0.1**2, (hidden_size, 1))
        self.b2 = np.random.normal(0.1, 0.1**2, (n_classes, 1))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        #changed: _, _, _, A2 = self.forward_prop(X)
        _, _, _, A2 = self.forward_prop(X.T)

        return A2.argmax(axis=0)
    
    def softmax(self, Z):
        #print("Output:", Z)
        #remove the max value from each column to avoid overflow
        Z_ = Z - np.max(Z, axis=0)
        e_Z = np.exp(Z_)
        # print("e_Z: ", e_Z)
        # print("e_Z.sum(axis=0): ", e_Z.sum(axis=0))
        return e_Z / e_Z.sum(axis=0)
    
    def forward_prop(self, X):
        #print("Shape of X in fp: ", X.shape)
        #print("Shape of W1: ", self.W1.shape)
        #exit(0)
        # print("Shape of W2: ", self.W2.shape)
        # print("Shape of b1: ", self.b1.shape)
        # print("Shape of b2: ", self.b2.shape)

        Z1 = self.W1.dot(X) + self.b1
        #changed_new: Z1 = self.W1.dot(X.T) + self.b1
        # ReLU activation function
        A1 = np.maximum(0, Z1)

        Z2 = self.W2.dot(A1) + self.b2
        # Softmax activation function
        A2 = self.softmax(Z2)

        return Z1, A1, Z2, A2
    
    #function that receives a 1xn vector of single digit integers and returns a vector with the one-hot encodings
    def one_hot(self, Y):
        if Y.shape:
            one_hot_y = np.zeros((self.W2.shape[0], Y.shape[0]))
            for i in range(Y.shape[0]):
                one_hot_y[Y[i]][i] = 1
        else:
            one_hot_y = np.zeros((self.W2.shape[0], 1))
            one_hot_y[Y] = 1

        return one_hot_y
    
    def derivative_ReLu(self, Z):
        return Z > 0

    def back_prop(self, Z1, A1, A2, W2, X, Y):
        
        one_hot_y = self.one_hot(Y)

        dZ2 = A2 - one_hot_y

        dW2 = dZ2.dot(A1.T)

        db2 = np.sum(dZ2)

        dZ1 = W2.T.dot(dZ2) * self.derivative_ReLu(Z1)

        dW1 = dZ1.dot(X.T)
        #changed_new: dW1 = dZ1.dot(X)

        db1 = np.sum(dZ1)

        return dW1, db1, dW2, db2

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        #print("Shape of X in evaluate: ", X.shape)
        y_hat = self.predict(X)
        #print(y_hat)
        #print(y_hat.shape)
        #exit(0)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        #print("Shape of X: ", X.shape)
        index = 0
        for x_i, y_i in zip(X, y):
            #print("Shape of x_i: ", x_i.shape)
            x_i = np.expand_dims(x_i, axis=1)
            
            #changed_new: x_i=x_i.T
            #print("Shape of x_i: ", x_i.shape)
            Z1, A1, Z2, A2 = self.forward_prop(x_i)
            # print("Shape of Z1: ", Z1.shape)
            # print("Shape of A1: ", A1.shape)
            # print("Shape of Z2: ", Z2.shape)
            # print("Shape of A2: ", A2.shape)

            dW1, db1, dW2, db2 = self.back_prop(Z1, A1, A2, self.W2, x_i, y_i)
            # print("Shape of dW1: ", dW1.shape)
            # print("Shape of db1: ", db1.shape)
            # print("Shape of dW2: ", dW2.shape)
            # print("Shape of db2: ", db2.shape)

            self.W1 -= learning_rate * dW1
            self.W2 -= learning_rate * dW2
            self.b1 -= learning_rate * db1
            self.b2 -= learning_rate * db2

            if index % 5000 == 0:
                print("Iteration: ", index)
                print("Accuracy: ", self.evaluate(X, y))
                #exit(0)
                # print("Z1: ", Z1)
                # print("A1: ", A1)
                # print("Z2: ", Z2)
                print("Probabilities: ", A2)

            index += 1

        #calculate loss
        #print("Shape of X outside: ", X.shape)
        _, _, _, A2 = self.forward_prop(X.T)
        one_hot_y = self.one_hot(y)
        #compute cross entropy loss given the probabilities and the one-hot encoding
        loss = -np.sum(one_hot_y * np.log(A2)) / X.shape[0]
        return loss

def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]
    #print(train_X.shape)
    #exit(0)
    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
