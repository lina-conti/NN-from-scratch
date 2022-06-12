from POSTclass import *
import pickle

usage = """ PART OF SPEECH TAGGING MLP

        """+sys.argv[0]+"""

"""

parser = argparse.ArgumentParser(usage = usage)
subparsers = parser.add_subparsers(dest='mode', help='train and test a model or test a pretrained one, consult the online help of each particular mode for more information')
train_parser = subparsers.add_parser("train")
train_parser.add_argument("train_file", type=str, help='file containing the training corpus in conllu format')
train_parser.add_argument("test_file", type=str, help='file containing the test corpus in conllu format')
train_parser.add_argument('-a', "--activation", default='relu', choices=["relu", "tanh", "sigmoid"], type=str, help='activation function to be used, default={\'relu\'}')
train_parser.add_argument('-l', "--learning_rate", default=0.1, type=float, help="learning rate to be used, default=0.1")
train_parser.add_argument('-s', "--sizes_hidden_layers", default=[24], type=list, help="list with the sizes of the hidden layers, default=[24] (a single hidden layer of 24 neurons)")
train_parser.add_argument('-m', "--embedding_size", default=40, type=int, help="size of the word embeddings, default=40")
train_parser.add_argument('-w', "--window_size", default=2, type=int, help="size of the window around the current word, default=2")
train_parser.add_argument('-b', "--batch_size", default=10, type=int, help="size of the mini batches, default=10")
train_parser.add_argument('-e', "--epochs", default= 100, type=int, help="number of training epochs, default=100")
test_parser = subparsers.add_parser("test")
test_parser.add_argument("test_file", type=str, help='file containing the test corpus in conllu format')
test_parser.add_argument("pickled_model", type=str, help='file containing the pickled model to be tested')
args = parser.parse_args()


if args.mode == 'train':
    list_activations = list(args.activation for h in args.sizes_hidden_layers)
    tagger = POSTagger(pathlib.Path(args.train_file), args.sizes_hidden_layers, list_activations, args.embedding_size, args.window_size, verbose = False)

    with Halo(text='training', spinner='dots'):
        tagger.fit(pathlib.Path(args.train_file), args.batch_size, args.learning_rate, args.epochs)

    print(f"Accuracy of the trained model on {args.test_file}: {tagger.test(pathlib.Path(args.test_file)) * 100}%")


if args.mode == 'test':
    with open(args.pickled_model, "rb") as openfile:
        tagger = pickle.load(openfile)

    print(f"Accuracy of {args.pickled_model} on {args.test_file}: {tagger.test(pathlib.Path(args.test_file)) * 100}%")
