#functions for preprocessing and pos tagger class
import pathlib
from collections import defaultdict
from random import sample
from NNclasses import *

class POSTagger:
    '''
    POS Tagger class.

    Contains all the methods needed for preprocessing the data and an MLP
    attribute to be trained for the POS tagging class.
    '''

    def __init__(self, training_path, list_sizes_hidden_layers, list_activations, embed_size, window_size, verbose = False):
        '''
        input:
            training_path = path in the format pathlib.Path, pointing to a .conllu file (or similar)
            list_sizes_hidden_layers = list of int with size of each hidden layer
            list_activations = list of strings of size the number of hidden layers
            embed_size = size of the embeddings
            window_size = size of the window of words around the current word to use as features
            verbose = to toggle debugging information
        '''
        self.verbose = verbose
        self.window_size = window_size

        _, train_words, train_tags = self.extract(training_path)
        self.train_vocab = train_words

        # i2w: list to go from a word id to the word
        # w2i: dictionary to go from a word to its id
        self.i2w, self.w2i = self.vocabulary(train_words, dummy = '<s>')
        # i2l: list to go from a label id to the label (pos tag)
        # w2i: dictionary to go from a label to its id
        self.i2l, self.l2i = self.vocabulary(train_tags)

        self.MLP = EmbeddingMLP(list_sizes_hidden_layers, list_activations, len(self.i2w), embed_size, window_size * 2 + 1, len(self.i2l), self.verbose)


    def get_sents(self, path):
        '''
        input: path in the format pathlib.Path, pointing to a .conllu file (or similar)
        output: list[list[list[string]]] with the content
            all data[sentences[words[conllu features]]]
        '''
        text = [line.split('\t') for line in
                path.read_text(encoding='utf-8').split('\n') if line if
                not line.startswith('#')]
        sents = []
        acc = []
        for i in range(0, len(text)):
            if text[i][0] == '1':
                sents.append(acc)
                acc = []
            acc.append(text[i])
        sents.append(acc)
        del (sents[0])
        return sents

    def get_obs(self, data):
        '''
        Takes as input the list of sentences (each sentence is a list of words, each word
        is a list of strings representing their features).
        Returns a list of tuples (words:str, pos_tags:str).

        input: list[list[list[string]]]
        output: list[tuple(list[str], list[str])
        '''
        all_obs = []
        word_counts = defaultdict(int)
        tag_counts = defaultdict(int)
        for sent in data:
            all_obs.append(([word[1] for word in sent], [word[4] for word in sent]))
            for word in sent:
                word_counts[word[1]]+=1
                tag_counts[word[4]]+=1

        return all_obs, word_counts, tag_counts

    def extract(self, path):
        '''
        Calls the two previous functions.

        input: pathlib.Path
        output: list[tuple(list[str], list[str])]
        '''
        sents = self.get_sents(path)
        return self.get_obs(sents)

    def vocabulary(self, symset, unknown = '<unk>', dummy = None):
        '''
        TODO
        '''
        symlist = list(set(symset))
        if unknown: symlist.append(unknown)
        if dummy: symlist.append(dummy)
        return symlist, {sym:idx for idx,sym in enumerate(symlist)}

    def prep_examples(self, data, training = False ):
        '''
        TODO
        '''
        word_counts = self.train_vocab
        X = []
        y = []
        for sentence, tagset in data:
            sentence = ['<s>']*self.window_size + sentence + ['<s>']*self.window_size
            tagset = ['<s>']*self.window_size + tagset + ['<s>']*self.window_size
            for i in range(self.window_size, len(sentence)-self.window_size):
                #convert words and pos tags to their ids
                temp = [word for word in sentence[i-(self.window_size):i+self.window_size+1]]
                for word in temp:
                    if training and word_counts[word] <= 1:
                        word = sample([word]+['<unk>'], 1)[0]
                    elif not training and word_counts[word] < 1:
                        word = '<unk>'
                X.append([self.w2i[w] if w in self.w2i else self.w2i['<unk>'] for w in temp ] )
                y.append(self.l2i[tagset[i]])
        return np.array(X), np.array(y)

    def fit(self, training_path, batch_size, learning_rate, epochs, dev_path = None):
        '''
        training_path = path in the format pathlib.Path, pointing to a .conllu file (or similar)
        batch_size: int
        learning_rate: float
        epochs: int
        Learns the parameters of the MLP on the training data passed to the function
        '''
        train, _, _ = self.extract(pathlib.Path(training_path))
        #TODO not cut train
        train_X, train_y = self.prep_examples(train[:50], training=True)
        dev_X, dev_y = [], []
        if dev_path:
            dev, _, _ = self.extract(pathlib.Path(dev_path))
            dev_X, dev_y = self.prep_examples(dev[20:], training = False)
        train_scores, dev_scores = self.MLP.fit(train_X, train_y, batch_size, learning_rate, epochs, 
                            dev_X, dev_y)
        return train_scores, dev_scores

    def predict(self, input_X):
        '''
        input_X: a matrix of size batch_size x input_size OR a vector of size input_size
        Output: the predicted class (index of the class) for each input (can be a single value or a vector)
        '''
        return self.MLP.predict(input_X)

    def test(self, test_path):
        '''
        test_path = path in the format pathlib.Path, pointing to a .conllu file (or similar)
        Output: the accuracy of the MLP for this test set
        '''
        test, _, _ = self.extract(pathlib.Path(test_path))
        test_X, test_y = self.prep_examples(test)
        return self.MLP.test(test_X, test_y)
