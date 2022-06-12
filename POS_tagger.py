from POSTclass import *

usage = """ PART OF SPEECH TAGGING MLP

        Trains a multilayer perceptron classifier with one hidden layer to solve the XOR problem.
        The default values are parameter values that we have generally found to work.
        They can be modified using the command line options.

        """+sys.argv[0]+""" [options]

"""

parser = argparse.ArgumentParser(usage = usage)
parser.add_argument('-a', "--activation", default='relu', type=str, help='Activation function to be used, must belong to {\'sigmoid\', \'tanh\', \'relu\'}. Default={\'relu\'}')
parser.add_argument('-l', "--learning_rate", default=0.1, type=float, help="Learning rate to be used. Default=0.01")
parser.add_argument('-s', "--hidden_size", default=24, type=int, help="Size of the hidden layer. Default=24")
parser.add_argument('-e', "--epochs", default= 1000, type=int, help="Number of training epochs. Default=1000")
parser.add_argument('-i', "--interactive_mode", action="store_true", help="Toggles the interactive mode, which takes as input to truth values and returns XOR of those.")
args = parser.parse_args()

tagger = POSTagger(pathlib.Path("surf.sequoia.train"), [48], ["relu"], 40, 2, verbose = False)

with Halo(text='training', spinner='dots'):
    tagger.fit(pathlib.Path("surf.sequoia.train"), 10, 0.1, 100)

print(tagger.test(pathlib.Path("surf.sequoia.test")))
