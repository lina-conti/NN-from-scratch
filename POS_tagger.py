#functions for preprocessing and pos tagger class 
from email.policy import default
import pathlib
from collections import defaultdict
from random import sample

def get_sents(path):
    '''input: path in the format pathlib.Path, pointing to a .conllu file
    output: list[list[list[string]]] with the content
        all data[sentences[words[conllu features]]]'''
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

def get_obs(data):
    '''takes as input the list of sentences (each sentence is a list of words, each word 
    is a list of strings representing their features). 
    returns a list of tuples (words:str, pos_tags:str)

        input: list[list[list[string]]]
        output: list[tuple(list[str], list[str])'''
    all_obs = []
    word_counts = defaultdict(int)
    tag_counts = defaultdict(int)
    for sent in data:
        all_obs.append(([word[1] for word in sent], [word[4] for word in sent]))
        for word in sent:
            word_counts[word[1]]+=1
            tag_counts[word[4]]+=1
        
    return all_obs, word_counts, tag_counts

def extract(path):
    '''calls the two previous functions. 
    input: pathlib.Path
    output: list[tuple(list[str], list[str])]'''
    sents = get_sents(path)
    return get_obs(sents)

def vocabulary(symset, unknown = '<unk>', dummy = None):
    symlist = list(set(symset))
    if unknown: symlist.append(unknown)
    if dummy: symlist.append(dummy)
    return symlist, {sym:idx for idx,sym in enumerate(symlist)}
    
def prep_examples(data, window_size, word_counts, w2i, l2i, training = False ):
    X = []
    y = []
    for sentence, tagset in data: 
        sentence = ['<s>']*window_size + sentence + ['<s>']*window_size
        tagset = ['<s>']*window_size + tagset + ['<s>']*window_size
        for i in range(window_size, len(sentence)-window_size):
            #convert words and pos tags to their ids
            temp = [word for word in sentence[i-(window_size):i+window_size+1]]
            for word in temp: 
                if training and word_counts[word] <= 1: 
                    word = sample([word]+['<unk>'], 1)[0]
                elif not training and word_counts[word] < 1: 
                    word = '<unk>'
            X.append([w2i[w] if w in w2i else w2i['<unk>'] for w in temp ] )
            y.append(l2i[tagset[i]])

    return X, y



