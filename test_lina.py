from NNclasses import *

mlp = MLP([2,2,1], ["sigmoid"])
print(mlp.forward_propagation([[1,1],[1,-1],[-1,1],[-1,-1]]))
