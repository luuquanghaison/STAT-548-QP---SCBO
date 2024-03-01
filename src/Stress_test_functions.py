# functions for 1d stress test
from Packages import *
from Objective_functions import *

def toy_1(X: torch.Tensor):
    """ 1d toy problem with obj and con optimum mismatched"""
    return toy_1d_objective(X, torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.1]))

def toy_2(X: torch.Tensor):
    """ 1d toy problem with obj and con optimum matched"""
    return toy_1d_objective(X, torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.1]))

def toy_3(X: torch.Tensor):
    """ 1d toy problem with easy con"""
    return toy_1d_objective(X, torch.tensor([0.5, 0.7]), torch.tensor([0.1, 2.0]))

def toy_4(X: torch.Tensor):
    """ 1d toy problem with adversarial obj """

    result = torch.rand(X.size(0),2)

    obj1 = toy_1d_objective(X, torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.1]))
    obj2 = toy_1d_objective(X, torch.tensor([0.7, 0.3]), torch.tensor([0.1, 0.1]))
    result[:,0] = obj1[:,0] - obj2[:,0]
    result[:,1] = obj1[:,1]

    return result

def toy_5(X: torch.Tensor):
    """ 1d toy problem with adversarial con """

    result = torch.rand(X.size(0),2)

    obj1 = toy_1d_objective(X, torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.1]))
    obj2 = toy_1d_objective(X, torch.tensor([0.7, 0.3]), torch.tensor([0.1, 0.1]))
    result[:,0] = obj1[:,0]
    result[:,1] = obj1[:,1] - obj2[:,1]

    return result

def toy_6(X: torch.Tensor):
    """ 1d toy problem with adversarial obj and con """

    result = torch.rand(X.size(0),2)

    obj1 = toy_1d_objective(X, torch.tensor([0.3, 0.7]), torch.tensor([0.1, 0.1]))
    obj2 = toy_1d_objective(X, torch.tensor([0.7, 0.3]), torch.tensor([0.1, 0.1]))
    result[:,0] = obj1[:,0] - obj2[:,0]
    result[:,1] = obj1[:,1] - obj2[:,1]

    return result

