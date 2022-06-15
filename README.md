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

The command line above will train a multilayer perceptron (MLP) classifier with one hidden layer to solve the XOR problem. The default hyperparameter values can be modified using the command line options. `-i` toggles the interactive mode that takes two boolean values as input and returns the XOR, as calculated by the trained model.

### POS Tagging

We then tested our MLP on the POS tagging task, which is more challenging, as not only weights and biases need to be trained, but also word embeddings.

#### `train` mode

```
$ python POS_tagger.py train [-h] [-a {relu,tanh,sigmoid}] [-l LEARNING_RATE] [-s SIZES_HIDDEN_LAYERS] [-m EMBEDDING_SIZE] [-w WINDOW_SIZE] [-b BATCH_SIZE] [-e EPOCHS] train_file dev_file test_file
```

This command will train an MLP on the train file, using the dev file for early stopping during training and calculate final trained accuracy on teh test file given as argument. The default hyperparameter values can be modified using the command line options.

#### `test` mode

```
$ python XOR_MLP.py [-h] test_file sequoia_model
```

This command can be used to test a model which was pre-trained on the [French sequoia corpus](https://deep-sequoia.inria.fr/) on the test_file passed as argument.
