import torch
import numpy as np
import zoopt
from zoopt.utils.zoo_global import nan


def Dixon_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x = x - 2
    D = len(random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    x1 = eff_dim[:, 0]
    xi = eff_dim[:, 1:]
    xi_minus_1 = eff_dim[:, :-1]
    f = (x1 - 1) ** 2 + ((torch.arange(2, D + 1) * (2 * xi ** 2 - xi_minus_1) ** 2).sum(dim=1))
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - (eff_dim ** 2).sum(dim=1)) / K
    return (f + dis)

# domain [-50, 50]
def griewank_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x - 10
    d_e = len(random_list)
    K = 10000
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2 / 4000
    a = a.sum(dim=1)
    b = torch.ones(x.shape[0])
    for i in range(d_e):
        b *= torch.cos(eff_dim[:, i] / torch.sqrt(torch.tensor(i + 1)))
    dis = x[:, :] ** 2
    dis1 = eff_dim[:, :] ** 2   
    dis = (dis.sum(dim=1) - dis1.sum(dim=1)) / K
    return (a - b + 1 + dis)

def levy_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x_eff = torch.index_select(x, 1, torch.tensor(random_list))
    w = 1 + (x - 1) / 4
    w = torch.index_select(w, 1, torch.tensor(random_list))
    a = torch.sin(torch.pi * w[:, 0]) ** 2
    c = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
    b = (w[:, :-1] - 1) ** 2 * (1 + 10*torch.sin(torch.pi * w[:, :-1] + 1) ** 2)
    b = b.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - (x_eff ** 2).sum(dim=1)) / K
    return  (a + b + c + dis)

# domain [-5.12, 5.12]
def sphere_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    x = x - 1
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return (a + dis)


# domain [-5, -10]
def rosenbrock_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x-1
    K = 10000
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    b = eff_dim[:, 1:] - eff_dim[:, :-1] ** 2
    b = 100 * (b ** 2)
    b = b + (eff_dim[:, 1:] - 1) ** 2
    b = b.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return (b + dis)

# [0, π]
def Michalewicz_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    m = 10
    x = x - 0.1
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    d_e = eff_dim.size(1)
    numOfSamples = x.shape[0]
    sum_term = torch.zeros(numOfSamples)
    for i in range(1, d_e + 1):
        xi = eff_dim[:, i - 1]
        new_term = torch.sin(xi) * torch.pow(torch.sin(i * xi ** 2 / np.pi), 2 * m)
        sum_term += new_term
    dis = (torch.sum(x**2, dim=1)-torch.sum(eff_dim**2, dim=1)) / K
    return (-sum_term + dis)


