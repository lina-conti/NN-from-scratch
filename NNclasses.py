import numpy as np
import random
import math
from scipy.special import softmax
import sys
import argparse
from halo import Halo
from utils import *

class Layer:
    '''
    Class containing what AffineLayer and ActivationLayer share in common
    It is also used to define the input layer simply
    '''

    def __init__(self, layer_size):
        '''
        layer_size: int
        Output: an instance of a Layer of layer_size neurons
        '''
        self.layer_size = layer_size
        # a matrix of size batch_size x layer_size
        # initialized and then updated by forward_propagation for AffineLayer and ActivationLayer
        # initialized manually for the input layer
        self.neuron_values = None

    def forward_propagation(self, batch_X):
        '''
        Abstract method to be defined in the subclasses
        batch_X: matrix (ndarray) of size batch_size x input_size
        Updates the attribute neuron_values
        '''
        pass

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        Abstract method to be defined in the subclasses
        values_previous_layer: the neuron values from the layer previous to this one in terms
             of forward propagation, a matrix of size
             batch_size x prev_layer
        layer_gradient: the gradient of the loss wrt this layer, calculated by the following layer,
            a matrix of size batch_size x layer_size
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation)
        '''
        pass

    def update(self, learning_rate):
        '''
        Abstract method to be defined in the subclasses
        learning_rate: float
        Updates the values of the parameters at the end of a back propagation on the whole NN
        '''
        pass


class AffineLayer(Layer):
    '''
    Affine layer of a neural network. Initialized with a number of neurons (should match the
    size of the preceding layer (input or activation layer)). Its parameters are randomly initialized.

    Forward propagation does a linear combination between the input and the weight matrix and adds the bias.
    Update modifies the values of the parameters (according to the previously computed gradients) and then zeroes the gradients.
    '''

    def __init__(self, input_size, layer_size):
        '''
        layer_size: int
        input_size: int
        Output: an instance of a AffineLayer with layer_size neurons and randomly initialized weights and biases
        '''
        super().__init__(layer_size)
        b = math.sqrt(6)/math.sqrt(input_size + layer_size)
        generator = np.random.default_rng()
        # matrix of size input_size x layer_size
        self.weights = generator.uniform(low=-b, high=b, size=(input_size, layer_size)) # as suggested by LaRochelle
        # vector of size layer_size
        self.bias = np.zeros(layer_size) # as suggested by LaRochelle

        # matrix of size input_size x layer_size (to be initialized during back propagation)
        self.weights_gradient = None
        # vector of size layer_size (to be initialized during back propagation)
        self.bias_gradient = None

    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix (ndarray) of size batch_size x input_size
        Updates the attribute neuron_values by doing a linear combination between the weights and the inputs in X and then summing the bias
        '''
        '''dot_porduct= np.dot(batch_X, self.weights) # batch_size x layer_size
        self.neuron_values=np.empty_like(dot_porduct) # batch_size x layer_size
        for i in range(len(dot_porduct)):
            self.neuron_values[i, :] = dot_porduct[i, :] + self.bias'''
        #print('affine layer: ', self.weights.shape, self.bias.shape, batch_X.shape)
        self.neuron_values = np.dot(batch_X, self.weights) + self.bias

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: the neuron values from the layer previous to this one in terms
             of forward propagation, a matrix of size
             batch_size x prev_layer
        layer_gradient: the gradient of the loss wrt this layer, calculated by the following layer,
            a matrix of size batch_size x layer_size
        Computes weights_gradient and bias_gradient
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation)
        '''
        # use outer product for the weights gradient
        # the gradients will have one dimension more, we need to "squeeze" them
        self.weights_gradient = np.dot(np.transpose(values_previous_layer), layer_gradient)

        self.bias_gradient = layer_gradient

        return np.dot(layer_gradient, np.transpose(self.weights))


    def update(self, learning_rate):
        '''
        learning_rate: float
        Updates the values of the parameters at the end of a back propagation on the whole NN
        Resets weights_gradient and bias_gradient to zeroes
        '''
        # Multiply weights_gradient by learning_rate
        # Substract it from weights
        # Proceed likewise for bias
        self.bias_gradient = np.sum(self.bias_gradient, axis = 0)
        self.bias = np.subtract(self.bias, (learning_rate*self.bias_gradient))
        #print('updated: ', self.bias.shape)
        self.weights= np.subtract(self.weights, (learning_rate*self.weights_gradient))



class ActivationLayer(Layer):
    '''
    Activation layer of a hidden layer in a neural network. Initialized with a number of neurons (should match the
    size of the preceding affine layer) and a custom activation function (sigmoid, tanh, relu)

    Forward propagation applies the activation function to the value at every neuron.
    Update does not do anything (there are no parameters to update).
    '''

    def __init__(self, layer_size, activation_function):
        '''
        layer_size: int
        activation_function: string id of an activation function to be fetched from the
            activation functions dictionary, must belong to {'sigmoid', 'tanh', 'relu'}
        Output: an instance of a ActivationLayer with layer_size neurons
        '''
        super().__init__(layer_size)
        #we use classes for the activation functions for better stability when pickling
        if activation_function == 'relu':
            self.activation_function = ReluActivation()
        elif activation_function == 'tanh':
            self.activation_function = TanhActivation()
        elif activation_function == 'sigmoid':
            self.activation_function = SigmoidActivation()
        else:
            print('error: no acceptable activation found')



    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix (ndarray) of size batch_size x input_size
        Updates the attribute neuron_values by applying the activation function to batch_X

        '''
        self.neuron_values = self.activation_function(batch_X)

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: the neuron values from the layer previous to this one in terms
             of forward propagation, a matrix of size batch_size x prev_layer
        layer_gradient: the gradient of this layer wrt. the loss, calculated at the layer After this one,
            a matrix of size batch_size x layer_size
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation),
            matrix of size previous_layer x batch_size
        '''
        grad = self.activation_function.deriv(values_previous_layer)
        return layer_gradient*grad

    def update(self, learning_rate):
        '''
        learning_rate: float
        doesn't do anything (there are no parameters to change in an activation layer)
        '''
        pass

class EmbeddingLayer(Layer):
    'the embedding layer for a mlp. stores words as randomly initialized vectors of a given length'

    def __init__(self, vocab_size, embed_size):
        '''randomly initialize a matrix of size vocab_size x embed_size'''
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.layer_size = embed_size
        #we use laRochell's strategy for initializing parameters 
        b = math.sqrt(6)/math.sqrt(vocab_size + embed_size)
        self.embedding_matrix = np.random.default_rng().uniform(low=-b, high=b, size=(vocab_size, embed_size)) 


    def forward_propagation(self, batch_X):
        '''given a batch of inputs, returns a batch of concatenations of the embeddings for each input'''
        self.input_ids = batch_X
        self.neuron_values = np.array([np.concatenate(
            [self.embedding_matrix[id] for id in ids_sequence]) for ids_sequence in batch_X])

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''Input: values_previous_layer (batch_size x num_words), the ids for the words in each batch
                  layer_gradient (batch_size x hidden_size), the gradient returned by layer H1

            Calculates the updates to the word embedding matrix

            Output: returns the values for the previous layer because i guess it had to return something'''
        # initialize gradient for word embeddings
        self.embeds_gradient = np.zeros((self.vocab_size, self.embed_size))
        # reshape input from batch_size x layer_size to batch_size x input_size x embed_size 
        # this gives us the gradients for individual words 
        shaped = layer_gradient.reshape(len(layer_gradient), len(values_previous_layer[0]), self.embed_size) # batch_size x num_words x embed size
        # init one-hot vectors so that the gradients go to the right embeddings
        one_hot = self.one_hot_matrix(values_previous_layer)
        # calculate the embeddings
        for i, o_h in enumerate(one_hot):
            update = np.dot(o_h.T, shaped[i])
            self.embeds_gradient += update

        return None # just so it returns something

    def one_hot_matrix(self, batch_values):
        '''helper function, gets the matrices of one-hot vectors for each batch elemement in backprop'''
        empty = np.zeros((len(batch_values), len(batch_values[0]), self.vocab_size)) #btch_size x w x V

        for i, in_seq in enumerate(batch_values):
            for j, val in enumerate(in_seq):
                empty[i,j, val]+=1
        return empty

    def update(self, learning_rate):
        self.embedding_matrix = np.subtract(self.embedding_matrix, learning_rate*self.embeds_gradient)


# AUXILIARY FUNCTIONS FOR MLP class
def get_one_hot_batch(batch_y, vector_size = None):
    '''
    batch_y: ndarray of size batch_size with the index of the gold class for each example in the batch
    Output: a batch of one-hot vectors with 1 at the y component for each example
    '''
    #print('shape ', batch_y.shape, 'size, ', batch_y.size)
    one_hot = np.zeros((batch_y.size, (vector_size if vector_size else batch_y.max()+1)))
    # an array of size batch_size with values from 0 to the bacth_size
    rows = np.arange(batch_y.size)
    # set the components at the index of the gold classes to 1
    one_hot[rows, batch_y] = 1
    return one_hot

class MLP:
    '''
    Multi-layer perceptron
    '''

    def __init__(self, list_sizes_hidden_layers, list_activations, input_size, number_of_classes, verbose = False):
        '''
        list_sizes_hidden_layers: list of int of size len(list_activations)
        list_activations: list of strings of size the number of hidden layers
        input_size: size of the input layer
        number_of_classes: C the size of the vocabulary of output classes
        verbose: to toggle additional information
        Output: an instance of MLP with layers_list initialized
        '''
        self.verbose = verbose
        # a list of Layer instances
        self.layers_list = []
        # add the input layer
        input_layer = Layer(input_size)
        self.layers_list.append(input_layer)

        size_previous_layer = input_size
        # add the hidden layers
        for i, h in enumerate(list_activations):    # i is the number of the hidden layer and h the activation function
            self.layers_list.append(AffineLayer(size_previous_layer, list_sizes_hidden_layers[i]))
            self.layers_list.append(ActivationLayer(list_sizes_hidden_layers[i],h))
            size_previous_layer = list_sizes_hidden_layers[i]

        # add the output layer (no activation, softmax is handled separately)
        self.layers_list.append(AffineLayer(list_sizes_hidden_layers[-1], number_of_classes))

    def __str__(self):
        '''stringify the object in some meaningful way'''
        output = ''
        for layer in self.layers_list:
            output+=f'{type(layer)}, {layer.layer_size}\n'
        return output

    def fit(self, training_X, training_y, batch_size, learning_rate, epochs,
                                dev_X = np.array([[]]), dev_y = np.array([[]]), patience = 10):
        '''
        training_X: matrix of size T x input_size
        training_y: a vector of size T
        batch_size: int
        learning_rate: float
        epochs: int
        Learns the parameters of the MLP on the training data passed to the function
        dev_X and dev_y: examples from the dev set. empty by default, if provided, used 
            to calculate early stopping to prevent overfitting. 
        patience: the patience interval for early stopping. 10 by default. 
        '''
        dev_scores = []
        train_scores = []

        # for each epoch
        for e in range(epochs):
            # shuffle
            examples = list(zip(training_X, training_y))
            random.shuffle(examples)
            training_X, training_y = zip(*examples)
            # separate into batches
            i = 0
            while i < len(examples):
                batch_X = np.array(training_X[i: i+batch_size])
                batch_y = np.array(training_y[i: i+batch_size])
                i += batch_size
                # for each batch, do forward, back propagation and update
                probabilities_output = self.forward_propagation(batch_X)
                if self.verbose: print('predicted: ', np.argmax(probabilities_output, axis = 1))
                self.back_propagation(probabilities_output, batch_y)
                self.update(learning_rate)
            train_scores.append(self.test(np.array(training_X), np.array(training_y)))
            if dev_X.any():#early stopping is enforced if there is a dev set in training
                dev_scores.append(self.test(dev_X, dev_y))
                #early stopping criteria: the most recent dev score must be better than at least one of the
                #past <patience> dev scores divided by .95 (a minimum improvement is required to keep training)
                if len(dev_scores)>patience and  dev_scores[-1] <= min(dev_scores[-patience:])/0.95:
                    print('Early stop at epoch', e)
                    return train_scores, dev_scores

            if  e%20==0:
                print('finished epoch ', e)
        return train_scores, dev_scores


    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix of size batch_size x input_size
        Loops through the layers calling forward propagation on each, then applies softmax and computes the loss
        Output: probabilities_output, a matrix of size batch_size x number_of_classes
        '''
        # initialize the input layer
        self.layers_list[0].neuron_values = batch_X
        # looping through the hidden layers, calling forward_propagation on each
        for k in range(1, len(self.layers_list)):
            if self.verbose:
                print(f'forward {k}: {len(self.layers_list[k-1].neuron_values)}')
            self.layers_list[k].forward_propagation(self.layers_list[k-1].neuron_values)
        # matrix of size batch_size x number_of_classes
        probabilities_output = softmax(self.layers_list[k].neuron_values, axis=1)
        return probabilities_output

    def back_propagation(self, probabilities_output, batch_y):
        '''
        probabilities_output: matrix of size batch_size x number_of_classes (result of forward_propagation)
        batch_y: vector of size batch_size
        Loops through the layers 'in reverse order' (wrt forward propagation) calling back_propagation on each
        '''
        # a batch of one-hot vectors with 1 at the y component for each example
        one_hot = get_one_hot_batch(batch_y, vector_size=len(probabilities_output[0]))
        if self.verbose:
            print('one-hots for output gradient: ', one_hot.shape)
        # gradient of NLL loss wrt to the pre-activation vectors at the output layer (=-e(y)-f(s) by LaRochelle's notations)
        output_gradient = - (one_hot - probabilities_output)
        # loop in reverse order through the layers (stopping before the input layer)
        layer_gradient = output_gradient
        for k in range(len(self.layers_list)-1, 0, -1):
            '''if self.verbose:

                print(f'layer {k}\n layer gradient: {layer_gradient}\n\
                     previous values{self.layers_list[k-1].neuron_values}')'''
            # back_propagation on each layer
            layer_gradient = self.layers_list[k].back_propagation(self.layers_list[k-1].neuron_values, layer_gradient)
            if self.verbose:
                if not k == 1:
                    print(f'gradient at {k}: {layer_gradient.shape}')
        return layer_gradient   #TODO why do we return this??

    def update(self, learning_rate):
        '''
        learning_rate: float
        Loops through the layers and calls their update method
        '''
        for layer in self.layers_list:
            layer.update(learning_rate)

    def predict(self, input_X):
        '''
        input_X: a matrix of size batch_size x input_size OR a vector of size input_size
        Output: the predicted class (index of the class) for each input (can be a single value or a vector)
        '''
        if input_X.ndim == 1:   # if we have a vector instead of a batch
            input_X = [input_X]
        # call forward_propagation and do an argmax
        scores = self.forward_propagation(input_X)
        return np.argmax(scores, axis = 1)

    def test(self, test_X, test_y):
        '''
        test_X: a matrix of size size_of_test_set x input_size
        test_y: a vector of size size_of_test_set
        Output: the accuracy of the MLP for this test set
        '''
        # call predict and compute the accuracy by comparing the results to test_y
        y_preds = self.predict(test_X)
        right = np.sum(y_preds == test_y)
        return right/len(test_y)


class EmbeddingMLP(MLP):
    def __init__(self, list_sizes_hidden_layers, list_activations, vocab_size, embed_size, input_size, number_of_classes, verbose = False):
        '''
        list_sizes_hidden_layers: list of int of size len(list_activations)
        list_activations: list of strings of size the number of hidden layers
        vocab_size: size of the vocabulary
        embed_size: size of the embeddings
        input_size: size of the input layer
        number_of_classes: C the size of the vocabulary of output classes
        verbose: to toggle additional information
        Output: an instance of EmbeddingMLP with layers_list initialized
        '''
        self.verbose = verbose
        # a list of Layer instances
        self.layers_list = []
        # add the input layer
        input_layer = Layer(input_size)
        self.layers_list.append(input_layer)
        # add the embedding layer
        embed_layer = EmbeddingLayer(vocab_size, embed_size)
        self.layers_list.append(embed_layer)

        size_previous_layer = embed_size * input_size
        # add the hidden layers
        for i, h in enumerate(list_activations):    # i is the number of the hidden layer and h the activation function
            self.layers_list.append(AffineLayer(size_previous_layer, list_sizes_hidden_layers[i]))
            self.layers_list.append(ActivationLayer(list_sizes_hidden_layers[i],h))
            size_previous_layer = list_sizes_hidden_layers[i]

        # add the output layer (no activation, softmax is handled separately)
        self.layers_list.append(AffineLayer(list_sizes_hidden_layers[-1], number_of_classes))
