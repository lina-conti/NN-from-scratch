import numpy as np
import random
import math
from scipy.special import softmax

class Layer:
    '''
    Abstract class containing what AffineLayer and ActivationLayer share in common
    '''

    def __init__(self, layer_size):
        '''
        layer_size: int
        Output: an instance of a Layer of layer_size neurons
        '''
        self.layer_size = layer_size
        # a matrix of size batch_size x layer_size (initialized and then updated by forward_propagation)
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
        values_previous_layer: TODO add description
        layer_gradient: TODO add description
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
    TODO add description
    '''

    def __init__(self, layer_size, input_size):
        '''
        layer_size: int
        input_size: int
        Output: an instance of a AffineLayer with layer_size neurons and randomly initialized weights and biases
        '''
        super().__init__(layer_size)
        # matrix of size input_size x layer_size
        self.weights = np.random.rand(input_size, layer_size) #randomly initialize them
        # vector of size layer_size
        self.bias = np.random.rand(layer_size)    # randomly initializes it

        # matrix of size input_size x layer_size (to be initialized during back propagation)
        self.weights_gradient = None
        # vector of size layer_size (to be initialized during back propagation)
        self.bias_gradient = None

    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix (ndarray) of size batch_size x input_size
        Updates the attribute neuron_values by doing a linear combination between the weights and the inputs in X and then summing the bias
        '''
        self.neuron_values = None  # batch_size x layer_size

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: TODO add description
        layer_gradient: TODO add description
        Computes weights_gradient and bias_gradient
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation)
        '''
        # use outer product for the weights gradient
        # the gradients will have one dimension more, we need to "squeeze" them
        pass

    def update(self, learning_rate):
        '''
        learning_rate: float
        Updates the values of the parameters at the end of a back propagation on the whole NN
        Resets weights_gradient and bias_gradient to zeroes
        '''
        # Multiply weights_gradient by learning_rate
        # Substract it from weights
        # Proceed likewise for bias
        pass

class ActivationLayer(Layer):
    '''
    TODO add description
    '''

    # define the activation function's dictionary here, so it is a "static" attribute of the class
    function_lookup = {
        'sigamoid': (lambda x: 1/(1+math.exp(x)), lambda x: x*(1-x)),
        'tanh': (lambda x: (math.exp(2*x)-1)/(math.exp(2*x)+1), lambda x: 1-x**2),
        'relu': (lambda x: max(0, x), lambda x: 1 if x>0 else 0)
    }

    def __init__(self, layer_size, activation_function):
        '''
        layer_size: int
        activation_function: string id of an activation function to be fetched from the activation function's dictionary
        Output: an instance of a ActivationLayer with layer_size neurons
        '''
        super().__init__(layer_size)
         # get activation function from the dictionary
        function, derivative = self.function_lookup[activation_function]

        self.activation_function = np.vectorize(function)
        self.derivative = np.vectorize(derivative)

    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix (ndarray) of size batch_size x input_size
        Updates the attribute neuron_values by applying the activation function to batch_X
        '''
        self.neuron_values= self.activation_function(batch_X)

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: TODO add description
        layer_gradient: TODO add description
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation)
        '''
        grad = self.derivative(values_previous_layer)
        return layer_gradient*grad

    def update(self, learning_rate):
        '''
        learning_rate: float
        Resets the gradients to zeroes (but will we store the gradients for activation layers?)
        '''
        pass

class MLP:
    '''
    Multi-layer perceptron
    '''

    def __init__(self, list_sizes_layers, list_activations):
        '''
        list_activations: list of strings of size the number of hidden layers
        list_sizes_layers: list of int of size len(list_activations) + 2 (for the input and output layers as well)
        Output: an instance of MLP with layers_list initialized
        '''
        # a list of Layer instances
        self.layers_list = []
        for i, h in enumerate(list_activations):    # i is the number of the layer and h the activation function
            self.layers_list.append(AffineLayer(list_sizes_layers[i],list_sizes_layers[i+1]))
            self.layers_list.append(ActivationLayer(list_sizes_layers[i+1],h))
        self.layers_list.append(AffineLayer(list_sizes_layers[i],list_sizes_layers[i+1]))   # output layer

    def fit(self, training_X, training_y, batch_size, learning_rate, epochs):
        '''
        training_X: matrix of size T x input_size
        training_y: a vector of size T
        batch_size: int
        learning_rate: float
        epochs: int
        Learns the parameters of the MLP on the training data passed to the function
        TODO: early stopping procedure (stop training when performance decreases on dev set)
        '''
        # for each epoch (for shuffling maybe use truc = zip(X,y), shuffle truc and then zip(*truc))
        for e in range(epochs):
            # shuffle
            examples = list(zip(training_X, training_y))
            random.shuffle(examples)
            training_X, training_y = zip(*examples)
            # separate into batches
            i = 0
            while i < len(examples):
                batch_X = training_X[i: i+batch_size]
                batch_y = training_y[i: i+batch_size]
                i += batch_size
                # for each batch, do forward, back propagation and update
                NLL_loss, probabilities_output = self.forward_propagation(batch_X)
                self.back_propagation(probabilities_output, batch_y)
                self.update(learning_rate)

    def forward_propagation(self, batch_X):
        '''
        batch_X: matrix of size batch_size x input_size
        Loops through the layers calling forward propagation on each, then applies softmax and computes the loss
        Output: probabilities_output, a matrix of size batch_size x number_of_classes
        '''
        self.layers_list[0].forward_propagation(batch_X)
        for i in range(1, len(self.layers_list)):
            self.layers_list[i].forward_propagation(self.layers_list[i-1].neuron_values)
        # matrix of size batch_size x number_of_classes
        probabilities_output = softmax(self.layers_list[i].neuron_values, axis=1)
        return probabilities_output

    def back_propagation(self, probabilities_output, batch_y):
        '''
        probabilities_output: matrix of size batch_size x number_of_classes (result of forward_propagation)
        batch_y: vector of size batch_size
        Loops through the layers 'in reverse order' (wrt forward propagation) calling back_propagation on each
        '''
        pass

    def update(self, learning_rate):
        '''
        learning_rate: float
        Loops through the layers and calls their update method
        '''
        pass

    def predict(self, input_X):
        '''
        input_X: a matrix of size batch_size x input_size OR a vector of size input_size
        Output: the predicted class (index of the class) for each input (can be a single value or a vector)
        '''
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
