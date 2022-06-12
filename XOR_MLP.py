from NNclasses import *

usage = """ XOR MULTILAYER PERCEPTRON CLASSIFIER

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

HIDDEN_SIZE = args.hidden_size
ACTIVATION = args.activation
LEARNING_RATE = args.learning_rate
EPOCHS = args.epochs

X_train = np.array([[1, 1],[1, 0],[0, 1],[0, 0]])
y_train = np.array([0, 1, 1, 0])


xor_mlp = MLP([HIDDEN_SIZE], [ACTIVATION], 2, 2, verbose = False)

with Halo(text='training', spinner='dots'):
    xor_mlp.fit(X_train, y_train, 4, LEARNING_RATE, EPOCHS)

if not args.interactive_mode:
    print("Train set inputs:\n", X_train)
    print("Train set gold classes:\n", y_train)
    print("Predictions of the trained model on the training set:\n", xor_mlp.predict(X_train))
else:
    x_1 = input("First truth value (type in 0 or 1): ")
    x_2 = input("Second truth value (type in 0 or 1): ")
    print(f"XOR({x_1},{x_2}) = {xor_mlp.predict(np.array([int(x_1), int(x_2)]))[0]}")
