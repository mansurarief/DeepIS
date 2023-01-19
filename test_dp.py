import random
import json
import numpy as np

from deepis import *

seed = 123
np.random.seed(seed)
random.seed(seed)

def f(x):
    return np.array(np.linalg.norm(x, axis=-1) > 5).reshape(-1,)

# collecting n1 samples
n1 = 2000 
dim = 2
X = np.random.uniform(0, 10, size=[n1, dim]) - 5
Y = f(X)
print(X.shape, Y.shape)
# training neural net
net = train_neural_network(X, Y, log=True, save=True, path="nets/net.pth")

#get neural net params
W = net.extract_params()

#search for dp
dp_opt = DeepISModel()
dp_opt.generate_opt_model(W)
dp_opt.model.pprint()
dp_opt.solve()

dp_list = [list(dp) for dp in dp_opt.dominating_points]

with open("results/dp_opt.json", "w") as outfile:
    json.dump(dp_list, outfile)

print("/nDone")
