#functions for preprocessing and pos tagger class 


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
    for sent in data:
        all_obs.append(([word[1] for word in sent], [word[3] for word in sent]))
    return all_obs

def extract(path):
    '''calls the two previous functions. 
    input: pathlib.Path
    output: list[tuple(list[str], list[str])]'''
    sents = get_sents(path)
    return get_obs(sents)

dev = extract('surf.sequoia.dev')
print(dev[0])