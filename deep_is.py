import random
import json


import numpy as np
import torch
import tqdm

from deepis import *

seed = 123
np.random.seed(seed)
random.seed(seed)
OUTPUT_INCREMENT = 10000
SAVE_EVERY = 10000

Problem = GaussianTailProbability(mu=[0, 0], sigma=0.5, rad=4.5)

# calculating ground truth value
print("Tail probability: {:.3e}".format(Problem.compute_target()))
timer_training = Timer()
timer_training.clock_in()
n1 = 2000  # initial sample
total_N = int(1e6)  # total number of sample
lb, ub = -5, 5  # lower- and upper-bound of the space

# collecting n1 samples
X = np.random.uniform(0, 10, size=[n1, Problem.dim]) - 5
Y = Problem.f(X)

# initiating grids
x1_grid = np.linspace(lb, ub, 400)
x2_grid = np.linspace(lb, ub, 400)
x1s, x2s = np.meshgrid(x1_grid, x2_grid)
xs = np.array([x1s.reshape(-1), x2s.reshape(-1)]).transpose()
ps = Problem.p.pdf(xs).reshape(x1s.shape)
ys = Problem.f(xs)

# training neural net
net = train_neural_network(X, Y, log=True, save=True, path="nets/net.pth")
print("\nTraining done")
def fhat(x): return np.diff(net(torch.Tensor(x)).cpu().data.numpy(), axis=-1) >= 0
timer_training.clock_out()

deepis_timer = Timer()
deepis_timer.clock_in()

# Deep IS
W = net.extract_params()
DeepISModelOpt = DeepISModel()
DeepISModelOpt.generate_opt_model(W)
DeepISModelOpt.solve()


DeepIS_proposal = PropDist(mu=DeepISModelOpt.dominating_points, sigma=Problem.sigma)
dp_list = [list(dp) for dp in DeepISModelOpt.dominating_points]

with open("results/DeepIS_DP.json", "w") as outfile:
    json.dump(dp_list, outfile)
    
deepis_timer.clock_out()


DeepIS_output = {}
for i in tqdm.trange(OUTPUT_INCREMENT, total_N-n1, OUTPUT_INCREMENT):
    iter_timer = Timer()
    iter_timer.clock_in()
    X_tilde = DeepIS_proposal.sample(i)
    Y_tilde = fhat(X_tilde).reshape(-1,1)
    likelihood_original = Problem.p.pdf(X_tilde).reshape(-1, 1)
    likelihood_proposal = DeepIS_proposal.pdf(X_tilde).reshape(-1, 1)
    DeepIS_est = Estimator(X_tilde, Y_tilde, likelihood_orig=likelihood_original,
                            likelihood_prop=likelihood_proposal, method='IS')
    iter_timer.clock_out()
    total_time = timer_training.total_duration + deepis_timer.total_duration + iter_timer.total_duration
    output = {
        'sample_size': i + n1,
        'mean': DeepIS_est.mean,
        'std': DeepIS_est.std,
        're': DeepIS_est.std / DeepIS_est.mean,
        'time': total_time.total_seconds()}
    DeepIS_output[str(i + n1)] = output

    if i % SAVE_EVERY == 0:
        with open("results/DeepIS_output.json", "w") as outfile:
            json.dump(DeepIS_output, outfile)

    
with open("results/DeepIS_output.json", "w") as outfile:
    json.dump(DeepIS_output, outfile)

print("/nDone")
