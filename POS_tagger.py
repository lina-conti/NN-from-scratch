from POSTclass import *
import jsonpickle
import json

usage = """ PART OF SPEECH TAGGING MLP

        """+sys.argv[0]+"""

"""

parser = argparse.ArgumentParser(usage = usage)
subparsers = parser.add_subparsers(dest='mode', help='train and test a model or test a pretrained one, consult the online help of each particular mode for more information')
train_parser = subparsers.add_parser("train")
train_parser.add_argument("train_file", type=str, help='file containing the training corpus in conll format')
train_parser.add_argument("dev_file", type=str, default=None, help='file containing the development corpus in conll format, default=None')
train_parser.add_argument("test_file", type=str, help='file containing the test corpus in conll format')
train_parser.add_argument('-a', "--activation", default='tanh', choices=["relu", "tanh", "sigmoid"], type=str, help='activation function to be used, default={\'tanh\'}')
train_parser.add_argument('-l', "--learning_rate", default=0.008, type=float, help="learning rate to be used, default=0.008")
train_parser.add_argument('-s', "--sizes_hidden_layers", default=[32], type=list, help="list with the sizes of the hidden layers, default=[32] (a single hidden layer of 32 neurons)")
train_parser.add_argument('-m', "--embedding_size", default=60, type=int, help="size of the word embeddings, default=60")
train_parser.add_argument('-w', "--window_size", default=4, type=int, help="size of the window around the current word, default=4")
train_parser.add_argument('-b', "--batch_size", default=10, type=int, help="size of the mini batches, default=10")
train_parser.add_argument('-e', "--epochs", default= 100, type=int, help="number of training epochs, default=100")
test_parser = subparsers.add_parser("test")
test_parser.add_argument("test_file", type=str, help='file containing the test corpus in conll format')
test_parser.add_argument("model", type=str, default='good_tagger.json', help='file containing the model to be tested in json format, default=\'good_tagger.json\'')
args = parser.parse_args()


if args.mode == 'train':
    list_activations = list(args.activation for h in args.sizes_hidden_layers)
    tagger = POSTagger(pathlib.Path(args.train_file), args.sizes_hidden_layers, list_activations, args.embedding_size, args.window_size,  verbose = False)

    with Halo(text='training', spinner='dots'):
        train_scores, dev_scores = tagger.fit(pathlib.Path(args.train_file), args.batch_size, args.learning_rate, args.epochs, dev_path=args.dev_file)

    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.DataFrame({'training set':train_scores, 'development set':dev_scores})
    df['epochs'] = df.reset_index().index
    df.plot(y=['training set','development set'], x='epochs', xlabel='Epoch', ylabel='Accuracy')
    plt.show()

    print(f"Accuracy of the trained model on {args.test_file}: {tagger.test(pathlib.Path(args.test_file)) * 100}%")

    with open("good_tagger.json", "w") as outfile:
        encoded_model = jsonpickle.encode(tagger)
        outfile.write(encoded_model)


if args.mode == 'test':
    with open(args.model, "r") as openfile:
        json_str = openfile.read()
        tagger = jsonpickle.decode(json_str)

    print(f"Accuracy of {args.model} on {args.test_file}: {tagger.test(pathlib.Path(args.test_file)) * 100}%")
