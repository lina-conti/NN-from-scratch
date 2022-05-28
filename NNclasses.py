import numpy as np
import random
import math
from scipy.special import softmax

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
        # TODO use better initialization values for the parameters?
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
        '''dot_porduct= np.dot(batch_X, self.weights) # batch_size x layer_size
        self.neuron_values=np.empty_like(dot_porduct) # batch_size x layer_size
        for i in range(len(dot_porduct)):
            self.neuron_values[i, :] = dot_porduct[i, :] + self.bias'''
        self.neuron_values = np.dot(batch_X, self.weights) + self.bias

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: the neuron values from the layer previous to this one in terms
             of forward propagation, can be a vector the size of the previous layer or a matrix of size
             batch_size x prev_layer
        layer_gradient: the gradient of the loss wrt this layer, calculated by the following layer,
            can be a vector of size layer_size or a matrix of size batch_size x layer_size
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
        self.bias = np.subtract(self.bias, (learning_rate*self.bias_gradient))
        self.weights= np.subtract(self.weights, (learning_rate*self.weights_gradient))


class ActivationLayer(Layer):
    '''
    Activation layer of a hidden layer in a neural network. Initialized with a number of neurons (should match the
    size of the preceding affine layer) and a custom activation function (sigmoid, tanh, relu)

    Forward propagation applies the activation function to the value at every neuron.
    Update does not do anything (there are no parameters to update).
    '''

    # define the activation function's dictionary here, so it is a "static" attribute of the class
    function_lookup = {
        'sigmoid': (lambda x: 1/(1+np.exp(x)), lambda x: x*(1-x)),
        'tanh': (lambda x: (np.exp(2*x)-1)/(np.exp(2*x)+1), lambda x: 1-x**2),
        'relu': (lambda x: max(0, x), lambda x: 1 if x>0 else 0)
    }

    def __init__(self, layer_size, activation_function):
        '''
        layer_size: int
        activation_function: string id of an activation function to be fetched from the
            layer's activation functions dictionary
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
        #test = np.array([-2, -1, 0, 1, 2, 3])
        self.neuron_values = self.activation_function(batch_X)

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''
        values_previous_layer: the neuron values from the layer previous to this one in terms
             of forward propagation. can be a vector the size of the previous layer or a matrix of size
             batch_size x prev_layer
        layer_gradient: the gradient of this layer wrt. the loss, calculated at the layer After this one
            can be a vector of size layer_size or a matrix of size batch_size x layer_size
        Output: the gradient of the previous layer (considering the order of the layers for forward propagation)
            vector of size previous_layer or matrix of size previous_layer x batch_size
        '''
        grad = self.derivative(values_previous_layer)
        return layer_gradient*grad

    def update(self, learning_rate):
        '''
        learning_rate: float
        doesn't do anything (there are no parameters to change in an activation layer)
        '''
        pass

class EmbeddingLayer(Layer): 
    'the embedding layer for a mlp. stores words as randomly initialized vectors of a given length'

    def __init__(self, vocab_size, embed_size, ones = False):
        '''randomly initialize a matrix of size vocab_size x embedding_size 
        TODO: smarter initialization'''
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        if ones: 
            self.weights = np.ones((vocab_size, embed_size))
        else:
            self.weights = np.random.rand(vocab_size, embed_size)
        print(self.weights)

    def forward_propagation(self, batch_X):
        '''given a batch of inputs, return the concatination of embeddings for each input'''
        self.in_ids = batch_X
        return np.array([np.concatenate([self.weights[ident] for ident in seq]) for seq in batch_X])

    def back_propagation(self, values_previous_layer, layer_gradient):
        '''input: values_previous layer (batch_size x window_size), the ids for the words in each batch
                    layer_gradient (batch_size x hidden_size), the gradient returned by layer H1
                    
            calculates the updates to the word embedding matrix 
            
            returns the values for the previous layer because i guess it had to return something'''
        #initialize gradient for word embeddings
        self.embeds_gradient = np.zeros((self.vocab_size, self.embed_size))
        #reshape input to embeddings for indivudual words
        shaped = layer_gradient.reshape(len(layer_gradient), len(values_previous_layer[0]), self.embed_size)#batch_size x num_words x embed size 
        #init one-hot vectors so that the gradients go to the right embeddings
        one_hot = self.one_hot_matrix(values_previous_layer)

        #calcualte the embeddings
        for i, o_h in enumerate(one_hot): 
            update = np.dot(o_h.T, shaped[i])
            self.embeds_gradient += update

        return values_previous_layer #this does nothing
        
    def one_hot_matrix(self, batch_values): 
        '''helper function, gets the matrices of one-hot vectors for each batch elemement in backprop'''
        empty = np.zeros((len(batch_values), len(batch_values[0]), self.vocab_size)) #btch_size x w x V
        print(empty)
        for i, in_seq in enumerate(batch_values): 
            for j, val in enumerate(in_seq):
                empty[i,j, val]+=1
        return empty

    def update(self, learning_rate):
        self.weights = np.subtract(self.weights, learning_rate*self.embeds_gradient)


# AUXILIARY FUNCTIONS FOR MLP class
def get_one_hot_batch(batch_y):
    '''
    batch_y: ndarray of size batch_size with the index of the gold class for each example in the batch
    Output: a batch of one-hot vectors with 1 at the y component for each example
    '''
    one_hot = np.zeros((batch_y.size, batch_y.max()+1))
    # an array of size batch_size with values from 0 to the bacth_size
    rows = np.arange(batch_y.size)
    # set the components at the index of the gold classes to 1
    one_hot[rows, batch_y] = 1
    return one_hot

class MLP:
    '''
    Multi-layer perceptron
    '''

    def __init__(self, list_sizes_layers, list_activations, verbose = False):
        '''
        list_activations: list of strings of size the number of hidden layers
        list_sizes_layers: list of int of size len(list_activations) + 2 (for the input and output layers as well)
        Output: an instance of MLP with layers_list initialized
        '''
        self.verbose = verbose
        # a list of Layer instances
        self.layers_list = []
        # add the input layer
        input_layer = Layer(list_sizes_layers[0])
        self.layers_list.append(input_layer)

        # add the hidden layers
        for i, h in enumerate(list_activations):    # i is the number of the layer and h the activation function
            self.layers_list.append(AffineLayer(list_sizes_layers[i],list_sizes_layers[i+1]))
            self.layers_list.append(ActivationLayer(list_sizes_layers[i+1],h))
        # add the output layer (no activation, softmax is handled separately)
        self.layers_list.append(AffineLayer(list_sizes_layers[-2],list_sizes_layers[-1]))   # output layer

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
                self.back_propagation(probabilities_output, batch_y)
                self.update(learning_rate)
            if self.verbose and e%5000==0: 
                print('finished epoch ', e)


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
        one_hot = get_one_hot_batch(batch_y)
        # gradient of NLL loss wrt to the pre-activation vectors at the output layer (=-e(y)-f(s) by LaRochelle's notations)
        output_gradient = - (one_hot - probabilities_output)
        # loop in reverse order through the layers (stopping before the input layer)
        layer_gradient = output_gradient
        for k in range(len(self.layers_list)-1, 0, -1):
            if self.verbose: 
                print(f'layer {k}\n layer gradient: {layer_gradient}\n\
                     previous values{self.layers_list[k-1].neuron_values}')
            # back_propagation on each layer
            layer_gradient = self.layers_list[k].back_propagation(self.layers_list[k-1].neuron_values, layer_gradient)
        return layer_gradient

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
