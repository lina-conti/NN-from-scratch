from POSTclass import *
import matplotlib
import matplotlib.pyplot as plt
import time

# Tunning the learning rate
possible_learning_rates = [0.008, 0.009, 0.01, 0.05]
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
possible_embed_sizes = [10, 20, 30, 40, 50, 60]
best_dev_acc_embeds = 0
best_size = None
all_best_accs_embeds = []
for size in possible_embed_sizes:
    with Halo(text = f'training with size = {size}', spinner = 'dots') as spinner:
        t = time.time()
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), [32], ['relu'], size, 2)
        train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 10, best_rate, 100, 'surf.sequoia.dev')
        if dev_scores[-1]>best_dev_acc_embeds:
            best_dev_acc_embeds = dev_scores[-1]
            best_size = size
            all_best_accs_embeds = dev_scores
    spinner.succeed(text=f'\n finished with {size}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}\
        , total training time = {time.time() - t}')
print(all_best_accs_embeds)
print(f'best size: {best_size}\nbest accuracy: {best_dev_acc_embeds}')

# Tunning the activation function
possible_activations = ['relu', 'tanh', 'sigmoid']
best_dev_acc_acts = 0
best_function = None
all_best_accs_acts = []
for act in possible_activations:
    with Halo(text = f'training with function = {act}', spinner = 'dots') as spinner:
        t = time.time()
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), [32], [act], best_size, 2)
        try:
            train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 
                                                10, best_rate, 100, 'surf.sequoia.dev')
            if dev_scores[-1]>best_dev_acc_acts:
                best_dev_acc_acts = dev_scores[-1]
                best_function = act
                all_best_accs_acts = dev_scores
        except: 
            print(f'error encountered training with {act} ')
    spinner.succeed(text=f'\n finished with {act}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}\
        , total training time = {time.time() - t}')
print(all_best_accs_acts)
print(f'best function: {best_function}\nbest accuracy: {best_dev_acc_acts}')


# Tunning the size and number of hidden layers
possible_layers = [[32], [32, 32], [64], [64, 32], [64, 64]]
best_dev_acc_layers = 0
best_setup = None
all_best_accs_layers = []
for setup in possible_layers:
    with Halo(text = f'training with layers = {setup}', spinner = 'dots') as spinner:
        t = time.time()
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), setup, [best_function]*len(setup), best_size, 2)
        try:
            train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 
                                                10, best_rate, 100, 'surf.sequoia.dev')
            if dev_scores[-1]>best_dev_acc_layers:
                best_dev_acc_layers = dev_scores[-1]
                best_setup = setup
                all_best_accs_layers = dev_scores
        except: 
            print(f'error encountered training with {setup} ')
    spinner.succeed(text=f'\n finished with {setup}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}\
        , total training time = {time.time() - t}')
print(all_best_accs_layers)
print(f'best layer setup: {best_setup}\nbest accuracy: {best_dev_acc_layers}')



# Tunning the window size
best_model = None
possible_window_sizes = [0, 1, 2, 3, 4]
best_dev_acc_windows = 0
best_window= None
all_best_accs_windows = []
for window in possible_window_sizes:
    with Halo(text = f'training with window = {window}', spinner = 'dots') as spinner:
        t = time.time()
        tagger = POSTagger(pathlib.Path('surf.sequoia.train'), 
            best_setup, [best_function]*len(best_setup), best_size, window)
        try:
            train_scores, dev_scores = tagger.fit(pathlib.Path('surf.sequoia.train'), 
                                                10, best_rate, 100, 'surf.sequoia.dev')
            if dev_scores[-1]>best_dev_acc_windows:
                best_dev_acc_windows = dev_scores[-1]
                best_window = window
                all_best_accs_windows = dev_scores
                best_model = tagger
        except: 
            print(f'error encountered training with {window} ')
    spinner.succeed(text=f'\n finished with {window}, dev acc {dev_scores[-1]}, total epochs {len(dev_scores)}\
        , total training time = {time.time() - t}')
print(all_best_accs_windows)
print(f'best window_size: {best_window}\nbest accuracy: {best_dev_acc_windows}')

print('best model: ', best_model)
print(f'best hyperparameters: \nLR = {best_rate}\nembed size = {best_size}\nactivation function = {best_function}\
    \nbest architecture = {best_setup}\nbest window = {best_window}')

print(f'best model accuracy on test: {best_model.test("surf.sequoia.test")}')
