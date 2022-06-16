# :brain: NN-from-scratch 
NLP Project 2022

## Presentation

This is an M1 university project in Natural Language Processing, by [Alice Bruguier](https://github.com/MichStrogoff), [Lina Conti](https://github.com/lina-conti) and [Isaac Murphy](https://github.com/isaac-murphy).

The goal was to implement a deep neural network (NN) from scratch. The NN was tested on the XOR problem and on a Part-of-Speech (POS) tagging task.  

A description of the project topic can be found [here](http://www.linguist.univ-paris-diderot.fr/~mcandito/projetsLI/projetsLI2122/en_m1_projects.htm).

The project was submitted in June 2022.


## Usage

### XOR Problem

We started by testing our NN on a simple problem, by trying to implement the exclusive or logic gate with it.

```
$ python XOR_MLP.py [-h] [-a {relu,tanh,sigmoid}] [-l LEARNING_RATE] [-s HIDDEN_SIZE] [-e EPOCHS] [-i]
```

The command line above will train a multilayer perceptron (MLP) classifier with one hidden layer to solve the XOR problem. `-i` toggles the interactive mode that takes two boolean values as input and returns the XOR, as calculated by the trained model.

The default hyperparameter values can be modified using the command line options:

- `activation_function`: activation function to be used, must belong to {'sigmoid', 'tanh', 'relu'}, default={'relu'}

- `learning_rate`: learning rate to be used, float, default=0.01

- `hidden_size`: size of the hidden layer, int, default=24

- `epochs`: maximal number of epochs to train for, int, default=1000


### POS Tagging

We then tested our MLP on the POS tagging task, which is more challenging, as not only weights and biases need to be trained, but also word embeddings.


#### `train` mode

```
$ python POS_tagger.py train [-h] [-a {relu,tanh,sigmoid}] [-l LEARNING_RATE] [-s SIZES_HIDDEN_LAYERS] [-m EMBEDDING_SIZE] [-w WINDOW_SIZE] [-b BATCH_SIZE] [-e EPOCHS] train_file dev_file test_file
```

This command will train an MLP on the train file, using the dev file for early stopping during training and calculate the accuracy of the trained model on the test file given as argument. 

- `train_file`: path to a file containing the training data in CoNLL format

- `dev_file`: path to a file containing the development set in CoNLL format

- `test_file`: path to a file containing the test data in CoNLL format

The default hyperparameter values can be modified using the command line options:

- `activation`: activation function to be used, must belong to {relu,tanh,sigmoid}, default='tanh'

- `learning_rate`: learning rate to be used, float, default=0.008

- `sizes_hidden_layers`: list with the sizes of the hidden layers (int), default=[32] (a single hidden layer of 32 neurons)

- `embedding_size`: size of the word embeddings, int, default=60

- `window_size`: size of the window around the current word to be used as features, int, default=4

- `batch_size`: size of the mini batches, int, default=10

- `epochs`: maximum number of training epochs, int, default=100


#### `test` mode

```
$ python XOR_MLP.py [-h] test_file sequoia_model
```

This command can be used to test a model which was pre-trained on the [French sequoia corpus](https://deep-sequoia.inria.fr/) on the test_file passed as argument.

- `test_file`: path to a file containing the test data in CoNLL format
