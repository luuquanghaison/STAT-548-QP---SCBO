# functions for BO results
from Packages import *
from Objective_functions import *
from BO_functions import *


def optimization_loop(train_X,T,obj_func,input_bounds,acq_fn_name,weight_samples,weight_update):
    """ Optimization loop for BO """
    # train_X: matrix of existing inputs
    # T: number of iterations
    # obj_func: objective and constraint function
    # acq_fn_name: acquisition function name
    # weight_samples: r x m Tensor of weight distribution
    # weight_update: function to update weight_samples

    train_y = observe(train_X,obj_func)
    feasible_count = feasible(train_y) # feasible points observed (n x m Tensor)

    # Optimization loop
    for i in tqdm.tqdm(range(T), desc="Optimization Iteration"):
        # Here's the typical optimization loop that we've seen in class a bunch of times

        # Step 0: compute the posterior of our GP surrogate model of the objective function
        gp = create_gp_conditioned_on_data(train_X, train_y)

        # Step 1: choose a new point to observe the objective function at
        X_i, acq_func = policy(gp,train_y,acq_fn_name,input_bounds,weight_samples)

        # Step 2: make an observation at the selected input
        y_i = observe(X_i,obj_func)

        # Step 3: update the dataset with our new input/observaiton pair
        train_X, train_y = update_dataset(train_X, train_y, X_i, y_i)

        # Step 4: update weight_samples
        s_w, feasible_count = weight_update(weight_samples, X_i, y_i, feasible_count,i)

    gp = create_gp_conditioned_on_data(train_X, train_y)
    return train_X, train_y, gp, s_w, feasible_count, acq_func



def feas_ratio_update(weight_samples, X_i, y_i, feasible_count, iter):
    """ Update weighting scheme based on faesible ratio """
    # weight_samples: current weight distribution
    # y_i: newly observed output
    # iter: current iteration
    # feasible_count: current feasible points observed
    n,m = y_i.shape
    d = X_i.shape[1]
    feasible_count_i = torch.zeros(n,m-1)
    # update feasible count
    for i in range(n):
        feasible_count_i[i,:] = (y_i[i,1:m]<=0)
    feasible_count = torch.cat((feasible_count,feasible_count_i),0)
    
    # update weight_samples
    for j in range(m-1):
        idx_r = weight_samples[:,j] > 0   # index of relax weights
        idx_t = weight_samples[:,j] < 0   # index of tighten weights
        acc_ratio = feasible_count[:,j].mean()
        if acc_ratio < 0.5:
            weight_samples[idx_r,j] = weight_samples[idx_r,j].mul((2)**(-1/d))    # less relax
            weight_samples[idx_t,j] = weight_samples[idx_t,j].mul((2)**(1/d))   # more tighten
        else:
            weight_samples[idx_r,j] = weight_samples[idx_r,j].mul((2)**(1/d))     # more relax
            weight_samples[idx_t,j] = weight_samples[idx_t,j].mul((2)**(-1/d))  # less tighten

    return weight_samples, feasible_count



def no_update(weight_samples, X_i, y_i, feasible_count, iter):
    """ Static weighting scheme """
    # weight_samples: current weight distribution
    # y_i: newly observed output
    # iter: current iteration
    # feasible_count: current feasible points observed
    n,m = y_i.shape
    feasible_count_i = torch.zeros(n,m-1)
    # update feasible count
    for i in range(n):
        feasible_count_i[i,:] = (y_i[i,1:m]<=0)
    feasible_count = torch.cat((feasible_count,feasible_count_i),0)
    
    return weight_samples, feasible_count



def feasible(train_y):
    """ Check dataset feasibility """
    n,m = train_y.shape
    feasible_vec = torch.zeros(n,m-1)
    for i in range(n):
        feasible_vec[i,:] = (train_y[i,1:m]<=0)
    return feasible_vec



def reward_and_feas_count(res: torch.Tensor, cond: torch.Tensor, min_val):
    """ Function for reward, constrained reward curves and feasible count """
    # res: matrix of function observations
    # cond: matrix of condition satisfaction indicator
    # min_val: default value when no feasible solution has been found
    T, n = res.shape # T: num of obs, n: num of experiment replica

    reward = torch.ones(T,n).mul(min_val)
    reward_ci = torch.zeros(T-1,3)

    con_reward = torch.ones(T,n).mul(min_val)
    con_reward_ci = torch.zeros(T-1,3)

    feas_count = torch.zeros(T,n)
    feas_count_ci = torch.zeros(T-1,3)

    for i in range(1,T):
        for j in range(n):
            if (res[i,j] > reward[i-1,j]):
                reward[i,j] = res[i,j]
            else:
                reward[i,j] = reward[i-1,j]

            if cond[i,j]:
                feas_count[i,j] = feas_count[i-1,j]+1
                if (res[i,j] > con_reward[i-1,j]):
                    con_reward[i,j] = res[i,j]
                else:
                    con_reward[i,j] = con_reward[i-1,j]
            else:
                feas_count[i,j] = feas_count[i-1,j]
                con_reward[i,j] = con_reward[i-1,j]

        reward_m = reward[i,:].mean(dim=0)
        reward_ci[i-1,0] = np.quantile(reward[i,:],0.25)
        reward_ci[i-1,1] = reward_m
        reward_ci[i-1,2] = np.quantile(reward[i,:],0.75)

        #con_reward_m = con_reward[i,:].mean(dim=0)
        con_reward_ci[i-1,0] = np.quantile(con_reward[i,:],0.25)
        con_reward_ci[i-1,1] = np.quantile(con_reward[i,:],0.5)
        con_reward_ci[i-1,2] = np.quantile(con_reward[i,:],0.75)

        feas_m = feas_count[i,:].mean(dim=0)
        feas_sd = feas_count[i,:].std(dim=0)
        feas_count_ci[i-1,0] = feas_m - feas_sd
        feas_count_ci[i-1,1] = feas_m
        feas_count_ci[i-1,2] = feas_m + feas_sd

    return reward, reward_ci, con_reward, con_reward_ci, feas_count, feas_count_ci





