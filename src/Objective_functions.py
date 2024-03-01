# Define objective functions
from Packages import *


def toy_1d_objective(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor):
    """ 1d toy problem:
            objective: a scaled and shifted normal pdf
            constraint: negative of a scaled and shifted normal pdf
    """
    # X: a [N x 1] matrix of inputs in the domain [0, 1]
    # mu: a 2 element vector of mean for obj and con functions
    # sd: a 2 element vector of standard deviation for obj and con functions
    # Return: a [N x 2] matrix of responses

    input_bounds = torch.tensor([[0.0], [1.0]])
    if X.dim() != 2 and X.size(-1) != 1:
        raise ValueError("X should be a n x 1 matrix!")
    if torch.any(X.lt(input_bounds[0])) or torch.any(X.gt(input_bounds[1])):
        raise ValueError("X should be in the domain [0, 1]")

    result = torch.rand(X.size(0),2)

    dist_obj = torch.distributions.Normal(mu[0],sd[0])
    scale_obj = dist_obj.log_prob(mu[0]).exp() - torch.min(dist_obj.log_prob(input_bounds[0]).exp(),dist_obj.log_prob(input_bounds[1]).exp())
    shift_obj = dist_obj.log_prob(mu[0]).exp().mul(2/scale_obj) - 1.3

    dist_con = torch.distributions.Normal(mu[1],sd[1])
    scale_con = dist_con.log_prob(mu[1]).exp() - torch.min(dist_con.log_prob(input_bounds[0]).exp(),dist_con.log_prob(input_bounds[1]).exp())
    shift_con = dist_con.log_prob(mu[1]).exp().mul(2/scale_con) - 1.3

    result[:,0] = dist_obj.log_prob(X[:,0]).exp().mul(2/scale_obj) - shift_obj
    result[:,1] = -dist_con.log_prob(X[:,0]).exp().mul(2/scale_con) + shift_con

    return result





def sim1(X: torch.Tensor):
    """ Simulation 1 in Gardner's paper """
    # X: a [N x 2] matrix of inputs in the domain [0, 6]^2
    # Return: a [N x 2] matrix of responses and constrains

    input_bounds = torch.tensor([[0.0, 0.0], [6.0, 6.0]])
    if X.dim() != 2 and X.size(-1) != 2:
        raise ValueError("X should be a n x 2 matrix!")
    if torch.any(X.lt(input_bounds[0,0])) or torch.any(X.gt(input_bounds[1,0])):
        raise ValueError("X should be in the domain [0, 6]^2")

    result = torch.rand(X.size(0),2)

    result[...,0] = -(torch.cos(2 * X[...,0]) * torch.cos(X[...,1]) + torch.sin(X[...,0]))
    result[...,1] = torch.cos(X[...,0] + X[...,1]) + 0.5

    return result




def sim2(X: torch.Tensor):
    """ Simulation 2 in Gardner's paper """
    # X: a [N x 2] matrix of inputs in the domain [0, 6]^2
    # Return: a [N x 2] matrix of responses and constrains

    input_bounds = torch.tensor([[0.0, 0.0], [6.0, 6.0]])
    if X.dim() != 2 and X.size(-1) != 2:
        raise ValueError("X should be a n x 2 matrix!")
    if torch.any(X.lt(input_bounds[0,0])) or torch.any(X.gt(input_bounds[1,0])):
        raise ValueError("X should be in the domain [0, 6]^2")

    result = torch.rand(X.size(0),2)

    result[...,0] = -(torch.sin(X[...,0]) + X[...,1])
    result[...,1] = 0.95 + torch.sin(X[...,0]) * torch.sin(X[...,1])

    return result




neg_hartmann6 = Hartmann(negate=True)
def con_hartmann(X: torch.Tensor):
    """ Negative Hartmann 6 function """
    # X: a [N x 6] matrix of inputs in the domain [0, 1]^6
    # Return: a [N x 6] matrix of responses and constrains

    input_bounds = torch.stack((torch.zeros(6),torch.ones(6)))
    if X.dim() != 6 and X.size(-1) != 6:
        raise ValueError("X should be a n x 6 matrix!")
    if torch.any(X.lt(input_bounds[0,0])) or torch.any(X.gt(input_bounds[1,0])):
        raise ValueError("X should be in the domain [0, 1]^6")

    result = torch.rand(X.size(0),2)

    result[...,0] = neg_hartmann6(X)
    result[...,1] = X.sum(dim=-1) - 1.5

    return result

