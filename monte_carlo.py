import random
import json


import matplotlib.patches as patches
import numpy as np

import tqdm

from deepis import *

seed = 123
np.random.seed(seed)
random.seed(seed)
OUTPUT_INCREMENT = 10000
SAVE_EVERY = 100000

Problem = GaussianTailProbability(mu=[0, 0], sigma=0.5, rad=4.5)

# calculating ground truth value
print("Tail probability: {:.3e}".format(Problem.compute_target()))

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

mc_output = {}
for i in tqdm.trange(OUTPUT_INCREMENT, total_N, OUTPUT_INCREMENT):
    iter_timer = Timer()
    iter_timer.clock_in()
    X_tilde = Problem.p.rvs(i)
    Y_tilde = Problem.f(X_tilde).reshape(-1,1)
    MC_est = Estimator(X_tilde, Y_tilde, method='MC')
    iter_timer.clock_out()
    total_time =  iter_timer.total_duration
    output = {
        'sample_size': i,
        'mean': MC_est.mean,
        'std': MC_est.std,
        're': MC_est.std / MC_est.mean,
        'time': total_time.total_seconds()}
    mc_output[str(i)] = output

    if i % SAVE_EVERY == 0:
        with open("results/MC_output.json", "w") as outfile:
            json.dump(mc_output, outfile)

print("/nDone")
