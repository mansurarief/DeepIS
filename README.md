# deepis

DeepIS is a python package that provides a neural network-based method for solving high-dimensional integration problems. The package includes a set of functions for training a neural network, extracting the network's parameters, and using these parameters to optimize a proposal distribution. The package also includes an estimator for computing the tail probability of a Gaussian distribution using the importance sampling method.

## Installation
To install deepis, `git clone`  from [DeepIS github repository](https://github.com/mansurarief/DeepIS.git).
```
git clone https://github.com/mansurarief/DeepIS.git
```

## Usage
Here is an example of how to use the deepis package to calculate the tail probability of a Gaussian distribution:

```python
import random
import json
import numpy as np
import torch
import tqdm
from deepis import *

#Setting constants
OUTPUT_INCREMENT = 1000
SAVE_EVERY = 1000


# Setting random seed
seed = 123
np.random.seed(seed)
random.seed(seed)

# Defining problem
Problem = GaussianTailProbability(mu=[0, 0], sigma=0.5, rad=4.5)

# Calculating ground truth value
print("Tail probability: {:.3e}".format(Problem.compute_target()))

# Collecting initial samples
n1 = 2000
total_N = int(1e4)
lb, ub = -5, 5
X = np.random.uniform(0, 10, size=[n1, Problem.dim]) - 5
Y = Problem.f(X)

# Training neural network
net = train_neural_network(X, Y, log=True, save=True, path="net.pth")
print("\nTraining done")
def fhat(x): return np.diff(net(torch.Tensor(x)).cpu().data.numpy(), axis=-1) >= 0

# Optimizing proposal distribution using DeepIS
W = net.extract_params()
DeepISModelOpt = DeepISModel()
DeepISModelOpt.generate_opt_model(W)
DeepISModelOpt.solve()
DeepIS_proposal = PropDist(mu=DeepISModelOpt.dominating_points, sigma=Problem.sigma)

# Estimating tail probability using importance sampling
DeepIS_output = {}
for i in range(OUTPUT_INCREMENT, total_N-n1, OUTPUT_INCREMENT):
    X_tilde = DeepIS_proposal.sample(i)
    Y_tilde = fhat(X_tilde).reshape(-1,1)
    likelihood_original = Problem.p.pdf(X_tilde).reshape(-1, 1)
    likelihood_proposal = DeepIS_proposal.pdf(X_tilde).reshape(-1, 1)
    DeepIS_est = Estimator(X_tilde, Y_tilde, likelihood_orig=likelihood_original,
                            likelihood_prop=likelihood_proposal, method='IS')
    output = {
        'sample_size': i + n1,
        'mean': DeepIS_est.mean,
        'std': DeepIS_est.std,
        're': DeepIS_est.std / DeepIS_est.mean
    }
    DeepIS_output[str(i + n1)] = output

    if i % SAVE_EVERY == 0:
        with open("DeepIS_output.json", "w") as outfile:
            json.dump(DeepIS_output, outfile)
```

The code will save the trained neural net at `net.pth` and output log at `DeepIS_output.json`.
