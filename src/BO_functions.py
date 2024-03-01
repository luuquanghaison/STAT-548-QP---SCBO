# BO functions
from Packages import *
from WCEI import *


def observe(X,obj_func):
    """ Observe a noisy version of the (true) objective function evaluated at X """
    # obj_func: a objective and constraint function
    # X: inputs
    # Return: a matrix of responses and constrains
    sigma = 0.0

    phi = obj_func(X)
    y = phi + sigma * torch.randn_like(phi)  # Additive i.i.d. N(0, sigma^2) noise
    return y



def create_gp_conditioned_on_data(train_X, train_y):
    """ Constructs a GP surrogate model that's conditioned on the dataset (train_X, train_Y) """
    # train_X: a matrix of inputs in our training set
    # train_y: a matrix of (noisy) observations that correspond to the train_X inputs

    # This GP will use
    # - A zero mean
    # - A Matern kernel

    sigma = 0.0

    gp = botorch.models.SingleTaskGP(
        train_X, train_y,  # initial data
        mean_module=gpytorch.means.ZeroMean(),  # Prior mean
        covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))  # Prior covariance
    )

    # Typically, we would learn the hyperparameters of our mean and covariance function
    # To keep things simple for this 1D toy demonstration, we'll lock them down
    gp.likelihood.initialize(noise=(sigma ** 2))  # Observational noise in our likelihood
    gp.covar_module.initialize(outputscale=1.)  # Covariance function outputscale
    gp.covar_module.base_kernel.initialize(lengthscale=0.2)  # Covariance function lengthscale

    return gp



def policy(gp,train_y,acq_fn_name,input_bounds,weight_samples):
    """Optimizes the acquisition function, and returns a new candidate."""
    # gp: the posterior GP model
    # train_y: sampled observations
    # acq_fn_name: acquisition function name
    # d: problem dimension
    # s_weight: sd of weight distribution

    if acq_fn_name == "CEI":
        if len(train_y[train_y[:,1]<=0.0,0]) < 1:
        #if len(train_y[(train_y[:,1]<=0.0) & (train_y[:,2]<=0.0),0]) < 1:
            best_f = -1000.0
        else:
            best_f = train_y[train_y[:,1]<=0.0,0].max()
        acq_func = botorch.acquisition.analytic.ConstrainedExpectedImprovement(
            model=gp, best_f=best_f, objective_index=0, constraints={x: (None, 0.0) for x in range(1,train_y.shape[1])}
        )
    elif acq_fn_name == "RCEI":
        if len(train_y[train_y[:,1]<=0.0,0]) < train_y.shape[1]-1:
            best_f = train_y[torch.argmax(train_y[:,1]),0].max()
        else:
            best_f = train_y[train_y[:,1]<=0.0,0].max()
        acq_func = botorch.acquisition.analytic.ConstrainedExpectedImprovement(
            model=gp, best_f=best_f, objective_index=0, constraints={x: (None, 0.0) for x in range(1,train_y.shape[1])}
        )
    elif acq_fn_name == "WCEI":
        best_f, indices = torch.sort(train_y[...,0],descending=True)
        best_f = torch.cat((best_f, torch.ones(1).mul(-1000.0)), 0)
        best_c = train_y[indices,...]
        best_c = best_c[...,range(1,best_c.shape[1])]
        acq_func = WeightedConstrainedEI(
            model=gp, best_f=best_f, best_c=best_c, raw_samples=1024,
            objective_index=0, constraints={x: (None, 0.0) for x in range(1,train_y.shape[1])},
            weight_samples=weight_samples
        )
    else:
        raise ValueError(acq_fn_name)



    # Find new X value that optimizes the acquisition function
    # BoTorch's optimize_acqf does a few fancy tricks under the hood to search the input space for
    #  for the best candidate. Increasing raw_samples and num_restarts should improve the quality of
    #  the search

    new_X, _ = botorch.optim.optimize_acqf(
        acq_function=acq_func,
        bounds=input_bounds,
        q=1,  # This means that we are only choosing 1 new X value per iteration
        raw_samples=512,  # How many candidates we should consider - sampled randomly from the input space
        num_restarts=2,  # How many sets of candidates we should consider (more is better)
    )

    return new_X, acq_func



def update_dataset(train_X, train_y, new_X, new_y):
    """Adds a new input/observation pair to the existing dataset of inputs/observations"""
    # train_X: matrix of existing inputs
    # train_y: matrix of existing noisy objective function observations at input locations
    # new_X: matrix with the new input
    # new_y: matrix with the new observed value

    train_X = torch.cat([train_X, new_X], dim=-2)
    train_y = torch.cat([train_y, new_y], dim=-2)
    return train_X, train_y