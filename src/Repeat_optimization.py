# Optimization loops for stress test
from Packages import *
from BO_functions import *
from BO_result_functions import *


def repeated_opt(obj,T,rep,acq_name,sd,update_fn,min_val,one_side = True,input_bounds = torch.tensor([[0.0], [1.0]])):
    """ Repeat optimization loops """
    # obj: objective function
    # T: number of iterations
    # rep: number of repetition
    # acq_name: acquisition function name
    # sd: control how relaxed or tightened WCEI is
    # update_fn: function to update weighting scheme
    # min_val: minimum obj value for reward curve
    # one_side: whether the weights are 1 sided
    # input_bounds: region of objective to optimize

    d = input_bounds.shape[1]

    res = torch.zeros(T+1,rep)
    cond = torch.zeros(T+1,rep)

    for iter in range(rep):
        initial_train_X = torch.rand(1, d)
        initial_train_y = observe(initial_train_X,obj)

        m = initial_train_y.shape[1]-1
        r = 1024
        engine = botorch.sampling.qmc.NormalQMCEngine(m,seed=1)
        weight_samples = engine.draw(r) # r x m
        for i in range(m):
            if one_side:
                weight_samples[:,i] = weight_samples[:,i].abs().mul(sd)
            else:
                weight_samples[:,i] = weight_samples[:,i].mul(sd)

        train_X, train_y, gp, ws, fc, acq_fn = optimization_loop(initial_train_X,T,obj,input_bounds,acq_name,weight_samples,update_fn)

        res[:,iter] = train_y[:,0]
        cond[:,iter] = fc.prod(dim=1)

    # compute reward and feasible proportion
    _, reward_ci, _, con_reward_ci,_ , feas_count_ci = reward_and_feas_count(res, cond, min_val)
    
    return reward_ci, con_reward_ci, feas_count_ci