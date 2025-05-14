import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import numpy as np

import warnings

warnings.filterwarnings('ignore')
torch.set_default_dtype(torch.double)


class BayesianOptimization:
    def __init__(self, dim, bounds, seed=0):
        self.dim = dim
        self.bounds = bounds
        self.seed = seed
        self.dataNum = 0
        self.bestSoFar = np.inf
        
        self.train_X = None
        self.train_Y = None
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def fitGP(self):
        m = self.train_Y.mean()
        s = self.train_Y.std() if self.train_Y.std() < 1e-6 else 1.
        
        self.gp = SingleTaskGP(self.train_X, (self.train_Y - m) / s)
        self.mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        # fit_gpytorch_mll(self.mll)
        fit_gpytorch_model(self.mll)
        return self.gp
    
    def generateCandidateUsingUCB(self, gp=None, q=1, num_restarts=5, raw_samples=20):
        beta = 0.2 * self.dim * np.log(2 * self.dataNum)
        if gp == None:
            UCB = UpperConfidenceBound(self.gp, beta=beta, maximize=False)
        else:
            UCB = UpperConfidenceBound(gp, beta=beta, maximize=False)
        try:
            candidate, acq_value = optimize_acqf(UCB, bounds=self.bounds, q=q, num_restarts=num_restarts, raw_samples=raw_samples)
        except:
            candidate = torch.rand(q, self.dim)
            acq_value = torch.zeros(q, 1)
        return candidate, acq_value
    
    def updateTrainingSet(self, candidate, candidate_value):
        if self.train_X == None:
            self.train_X = candidate
            self.train_Y = candidate_value
        else:
            self.train_X = torch.cat([self.train_X, candidate])
            self.train_Y = torch.cat([self.train_Y, candidate_value])
        self.dataNum = self.train_X.shape[0]
        self.bestSoFar = self.train_Y.min()
        return self.train_X, self.train_Y
    
    def getDataNum(self):
        return self.dataNum
    
    def getBestSoFar(self):
        return self.bestSoFar
    
