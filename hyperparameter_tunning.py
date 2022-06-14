from POSTclass import *
import matplotlib
import matplotlib.pyplot as plt

# Tunning the learning rate
possible_learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
best_dev_acc = 0
best_rate = None
all_best_accs = []
for rate in possible_learning_rates:
    with Halo(text = f'training at {rate}', spinner = 'dots') as spinner:
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), [32], ['relu'], 40, 2)
        train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 10, rate, 100, 'surf.sequoia.dev')
        if dev_scores[-1]>best_dev_acc:
            best_dev_acc = dev_scores[-1]
            best_rate = rate
            all_best_accs = dev_scores
    spinner.succeed(text=f'\n finished with {rate}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}')
print(all_best_accs)
print(f'best rate: {best_rate}\nbest accuracy: {best_dev_acc}')


# Tunning the size of the embeddings
possible_learning_rates = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
best_dev_acc = 0
best_rate = None
all_best_accs = []
for rate in possible_learning_rates:
    with Halo(text = f'training at {rate}', spinner = 'dots') as spinner:
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), [32], ['relu'], 40, 2)
        train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 10, rate, 100, 'surf.sequoia.dev')
        if dev_scores[-1]>best_dev_acc:
            best_dev_acc = dev_scores[-1]
            best_rate = rate
            all_best_accs = dev_scores
    spinner.succeed(text=f'\n finished with {rate}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}')
print(all_best_accs)
print(f'best rate: {best_rate}\nbest accuracy: {best_dev_acc}')

# Tunning the activation function

# Tunning the size of hidden layers

# Tunning the number of the hidden layers

# Tunning the window size
