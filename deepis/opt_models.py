from multiprocessing.sharedctypes import Value
from typing import List

import numpy as np

import pyomo.environ as pyomo
from pyomo.opt import SolverFactory


class DeepISModel():
    def __init__(self, num_layers: int = 2, dims: List = [2, 10, 2]) -> None:
        self.num_layers = num_layers
        self.dimensions = dims
        self.layer = [str(i) for i in range(num_layers)]
        self.input_dim = [str(i) for i in range(dims[0])]
        self.s1_dim = [str(i) for i in range(dims[1])]
        self.s2_dim = [str(i) for i in range(dims[2])]
        self.bigM = 100000
        self.mu = {str(i): 0 for i in range(dims[0])}

    def generate_opt_model(self, W_dict):
        model = pyomo.ConcreteModel()

        # set definition
        model.layer = pyomo.Set(initialize=self.layer)
        model.input_dim = pyomo.Set(initialize=self.input_dim)
        model.s1_output_dim = pyomo.Set(initialize=self.s1_dim)
        model.s2_output_dim = pyomo.Set(initialize=self.s2_dim)

        # variable declaration
        model.s0 = pyomo.Var(model.input_dim, domain=pyomo.Reals)
        model.s1 = pyomo.Var(model.s1_output_dim, domain=pyomo.NonNegativeReals)
        model.z1 = pyomo.Var(model.s1_output_dim, domain=pyomo.Binary)
        model.s2 = pyomo.Var(model.s2_output_dim, domain=pyomo.NonNegativeReals)

        # parameters
        model.weights = pyomo.Param(model.layer, within=pyomo.Any, initialize=W_dict)
        model.mu = pyomo.Param(model.input_dim, within=pyomo.Reals, initialize=self.mu)
        model.bigM = pyomo.Param(initialize=10000)

        # layer 1
        def cons2(model, s1_output_dim):
            s1_hat = sum(model.weights['0']['weight'][s1_output_dim][i] * model.s0[i]
                         for i in model.input_dim) + model.weights['0']['bias'][s1_output_dim]
            return model.s1[s1_output_dim] <= s1_hat + model.bigM * (1 - model.z1[s1_output_dim])
        model.ineq1 = pyomo.Constraint(model.s1_output_dim, rule=cons2, doc='ReLU constrain1')

        def cons3(model, s1_output_dim):
            s1_hat = sum(model.weights['0']['weight'][s1_output_dim][i] * model.s0[i]
                         for i in model.input_dim) + model.weights['0']['bias'][s1_output_dim]
            return model.s1[s1_output_dim] >= s1_hat
        model.ineq2 = pyomo.Constraint(model.s1_output_dim, rule=cons3, doc='ReLU constrain2')

        def cons4(model, s1_output_dim):
            return model.s1[s1_output_dim] <= model.bigM * model.z1[s1_output_dim]
        model.ineq3 = pyomo.Constraint(model.s1_output_dim, rule=cons4, doc='ReLU constrain3')

        # layer 2
        def cons5(model, s2_output_dim):
            s2_hat = sum(model.weights['1']['weight'][s2_output_dim][i] * model.s1[i]
                         for i in model.s1_output_dim) + model.weights['1']['bias'][s2_output_dim]
            return model.s2[s2_output_dim] == s2_hat
        model.eq2 = pyomo.Constraint(model.s2_output_dim, rule=cons5, doc='Output constrain')

        def cons6(model):
            return model.s2['1'] - model.s2['0'] >= 0
        model.ineq4 = pyomo.Constraint(rule=cons6, doc='Positive prediction')

        # objective function
        def obj(model):
            #obj_val = (model.s0['0']-model.mu['0'])**2 + (model.s0['1']-model.mu['1'])**2
            obj_val = (model.s0['0'] - model.mu['0']) + (model.s0['1'] - model.mu['1'])
            return obj_val
        model.obj = pyomo.Objective(rule=obj, sense=pyomo.minimize)
        self.model = model

        return model

    def solve(self, solver: str = 'gurobi') -> None:
        self.solver = solver
        self.optimizer = pyomo.SolverFactory(solver)
        self.optimizer.options['NonConvex'] = 2
        self._solve()
        self._iterative_cutting_plane()

    def _solve(self, print_result: bool = False) -> None:
        results = self.optimizer.solve(self.model)
        self.res_status = results.solver.status
        if print_result:
            print('{} -- Obj: {:.2f}, x*: [{}]'.format(self.res_status,
                  self.model.obj(), [self.model.s0[i]() for i in self.model.input_dim]))

        self.dominating_points = None
        if self.res_status == 'ok':
            Xsol = np.array([[self.model.s0[i]() for i in self.model.input_dim]])
            self.dominating_points = Xsol

    def _iterative_cutting_plane(self, print_result: bool = False):
        model = self.model

        # initializing solving parameters
        self.cut_iter_max = 100  # maximum number of dominating points
        self. eps_ignore = 1e-1  # small value ==> Ax + b < 0 ---> Ax + b <= 0 - eps_ignore
        iter_ = 1
        optimizer = self.optimizer

        # initialize cutting planes constraint list
        model.cuts = pyomo.ConstraintList()

        # adding the first cutting plane
        expr = self.eps_ignore + sum((model.s0[i]() - model.mu[i]) *
                                     (model.s0[i] - model.s0[i]()) for i in model.input_dim)
        model.cuts.add(expr <= 0)

        # results.write() #print the result if necessary

        # initializing the solution array, assuming the problem is solvable
        Xsol = np.empty(shape=[0, self.dimensions[0]])
        if self.res_status == 'ok':
            Xsol = self.dominating_points

        # iterating, stop if the problem becomes infeasible
        while iter_ <= self.cut_iter_max and self.res_status == 'ok':
            iter_ += 1
            results = optimizer.solve(model)
            self.res_status = results.solver.status

            # if still solvable, add new cutting plane
            if self.res_status == 'ok':
                expr = self.eps_ignore + sum((model.s0[i]() - model.mu[i]) *
                                             (model.s0[i] - model.s0[i]()) for i in model.input_dim)
                model.cuts.add(expr <= 0)

                # storing and printing the solution
                Xsol = np.vstack((Xsol, np.array([[model.s0['0'](), model.s0['1']()]])))
                if print_result:
                    print('Obj: {:.2f}, x*: [{:.2f}, {:.2f}]'.format(model.obj(), model.s0['0'](), model.s0['1']()))

        self.dominating_points = Xsol


