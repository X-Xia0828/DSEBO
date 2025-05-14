import torch
import numpy as np
import os
import argparse
import random
import utils.functions as tfs
import utils.BayesOptimizer as bo
import utils.RandomEmbeddingGroup as reg


torch.set_default_dtype(torch.double)


## command line arguments
parser = argparse.ArgumentParser()
# arguments about the task
parser.add_argument('--seed', type=int, default=1, nargs='?')
parser.add_argument('--objDim', type=int, default=1000, nargs='?')
parser.add_argument('--effDim', type=int, default=30, nargs='?')
parser.add_argument('--budget', type=int, default=500, nargs='?')
parser.add_argument('--objectiveFunction', type=str, default='Sphere', nargs='?')
# arguments about the policy
parser.add_argument('--dHigh', type=int, default=100, nargs='?')
parser.add_argument('--dLow', type=int, default=5, nargs='?')
parser.add_argument('--beta', type=float, default=12, nargs='?')
args = parser.parse_args()

# set random seed
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)

## ABOUT THE OBJECTIVE FUNCTION
# set up objective function parameters
budget = args.budget
objDim = args.objDim
effDim = args.effDim

dHigh = args.dHigh
dLow = args.dLow
beta = args.beta
random.seed(0)
idx = random.sample([dim for dim in range(objDim)], effDim)
idx.sort()
objectiveFunc = args.objectiveFunction

# set up objective function
if args.objectiveFunction == 'Sphere':
    objectiveFunction = tfs.Sphere(dim_high=objDim, rand_list=idx)
elif args.objectiveFunction == 'Levy':
    objectiveFunction = tfs.Levy(dim_high=objDim, rand_list=idx)
elif args.objectiveFunction == 'Rosenbrock':
    objectiveFunction = tfs.Rosenbrock(dim_high=objDim, rand_list=idx)
elif args.objectiveFunction == 'Griewank':
    objectiveFunction = tfs.Griewank(dim_high=objDim, rand_list=idx)
elif args.objectiveFunction == 'Dixon':
    objectiveFunction = tfs.Dixon(dim_high=objDim, rand_list=idx)
elif args.objectiveFunction == 'Michalewicz':
    objectiveFunction = tfs.Michalewicz(dim_high=objDim, rand_list=idx)
bounds = torch.stack([torch.ones(objDim) * (-1), torch.ones(objDim)])

## ABOUT THE POLICY
# Dataset of every query in objective space
dataset_X = torch.empty(0, objDim)
dataset_Y = torch.empty(0, 1)
optimizerNos = []  # the dimension No. of optimizer of every iteration

## define of every low dimension optimizer, random embedding and data record
class OptDimension():
    def __init__(self, dim, optBounds=None, seed=0):
        self.dim = dim  # (integer)
        self._optBounds = optBounds if optBounds != None else torch.stack([torch.ones(dim) * (-1), torch.ones(dim)])
        self._opter = bo.BayesianOptimization(dim, self._optBounds, seed=seed)
        self.optQueries_X = torch.empty(0, dim)
        self.optQueries_Y = torch.empty(0, 1)
        self.queriesTimes = 0  # ONLY use this dim to query and update
        self.dataNum = 0  # every update include using other dim data
        self.bestSoFar = np.inf  # for minimization problem
        self.useLow = False  # whether to use low dimension data when first queries
        self.seed = seed
    
    def registerQuery(self, xLow, y):
        xLow = xLow.reshape(1, -1)
        self.optQueries_X = torch.cat([self.optQueries_X, xLow])
        self.optQueries_Y = torch.cat([self.optQueries_Y, y])
        self._opter.updateTrainingSet(xLow, y.reshape(1, -1))
        self._opter.fitGP()
        self.queriesTimes += 1
        self.dataNum += 1
        self.bestSoFar = min(self.bestSoFar, float(y))
    
    def acquisition(self):
        nextLow, _ = self._opter.generateCandidateUsingUCB()
        return nextLow
    
    def getQueries(self):
        return self.optQueries_X, self.optQueries_Y
    
    def initWithLowdimData(self, optQueries_X, optQueries_Y):
        self.optQueries_X = optQueries_X
        self.optQueries_Y = optQueries_Y
        self._opter.updateTrainingSet(optQueries_X, optQueries_Y)
        self._opter.fitGP()
        self.dataNum = self.optQueries_X.shape[0]
        self.bestSoFar = float(self.optQueries_Y.min())
        print(f"[Dim {self.dim}] Initialized with {self.dataNum} low dimension data!")
        
    def updateBounds(self, optBounds=None):
        self._optBounds = optBounds if optBounds != None else self._optBounds * 0.8
        self._opter = bo.BayesianOptimization(self.dim, self._optBounds, seed=seed)
        self._opter.updateTrainingSet(self.optQueries_X, self.optQueries_Y)
        self._opter.fitGP()
        print(f"=====\nUpdate the bounds of dim {self.dim} to {self._optBounds}\n=====")


# directory to save the results
directory_path = f"./results/{objectiveFunc}-{budget}/{objDim}-{effDim}/{beta}/"
os.makedirs(directory_path, exist_ok=True)  # check if the folder exists


######################
## THE MAIN PROCESS ##
######################

query = 0  # current query time
dims = dict()  # every dimension explored
speeds = []

## 1. initialize the embeding module
embedder = reg.RandomEmbeddingGroup(objDim, dHigh, bounds, seed=seed)
currentDim = dLow
jumpTime = int(budget / 2 / beta)
bsf = np.inf
bsfTime = 0
bestSoFar = []
while query < budget:
    ## 2. use old data to initialize the higher optimizer
    if currentDim not in dims.keys():
        dims[currentDim] = OptDimension(currentDim, seed=seed)
        if currentDim == dLow:
            xLow = torch.rand(1, currentDim) * 2 - 1
            xHigh = embedder(currentDim, xLow)
            y = torch.tensor(objectiveFunction.evaluate_true(xHigh)).reshape(1, -1)
            dims[currentDim].registerQuery(xLow, y)
            dataset_X = torch.cat([dataset_X, xHigh])
            dataset_Y = torch.cat([dataset_Y, y])
            for _ in range(y.shape[0]):
                bestSoFar.append(round(dataset_Y.min().item(), 4))
            bsf = float(y)
            bsfTime += 1
            query += 1
            print(f"[Query {query}] Obj.Val: {float(y):.4f}, bsf: {float(dataset_Y.min()):.4f}, curr dim {currentDim}, {bsf:.4f} bsfTime {bsfTime}")
        elif dims[currentDim].dataNum == 0:
            dimsList = list(dims.keys())
            dLast = sorted(dimsList)[-2]
            train_Xs, train_Ys = dims[dLast].getQueries()
            
            queryLowData = int(len(train_Xs))
            indices = torch.randperm(len(train_Xs))[:queryLowData]
            train_Xs_sample = train_Xs[indices]
            train_Ys_sample = train_Ys[indices]
            
            train_Xs_sample = embedder.dimProjection(dLast, currentDim, train_Xs_sample)  # project the data to current dim
            dims[currentDim].initWithLowdimData(train_Xs_sample, train_Ys_sample)  # use these data to init the optimizer
            
            print(f"=== use {queryLowData} queries on dim {dLast} data to init dim {currentDim}")
    
    ## 3. optimize the current dimension
    xLow = dims[currentDim].acquisition()
    xHigh = embedder(currentDim, xLow)
    y = torch.tensor(objectiveFunction.evaluate_true(xHigh)).reshape(1, -1)
    dims[currentDim].registerQuery(xLow, y)
    dataset_X = torch.cat([dataset_X, xHigh])
    dataset_Y = torch.cat([dataset_Y, y])
    for _ in range(y.shape[0]):
        bestSoFar.append(round(dataset_Y.min().item(), 4))
    optimizerNos.append(currentDim)
    query += 1
    bsfTime += 1
    print(f"[Query {query}] Obj.Val: {float(y):.4f}, bsf: {float(dataset_Y.min()):.4f}, curr dim {currentDim}, {bsf:.4f} bsfTime {bsfTime}")
    
    ## 4. judge bsf and bsf time
    if bsf - float(dataset_Y.min()) > 5e-1:
        bsf = min(float(dataset_Y.min()), bsf)
        bsfTime = 0
    
    ## 5. judge weather to update the dimension
    if bsfTime >= jumpTime:
        if currentDim == dHigh:
            deltaDim = 0
        elif len(speeds) < 2:
            deltaDim = int((dHigh - dLow) * 2 / beta)
            dimsList = list(dims.keys())
            if len(dimsList) > 1:
                dLast = sorted(dimsList)[-2]
                currSpeed = - (dims[currentDim].bestSoFar - dims[dLast].bestSoFar) / (currentDim - dLast)
                speeds.append(currSpeed)
        else:
            dimsList = list(dims.keys())
            dLast = sorted(dimsList)[-2]
            currSpeed = - (dims[currentDim].bestSoFar - dims[dLast].bestSoFar) / (currentDim - dLast)
            minSpeed = min(currSpeed, min(speeds))
            maxSpeed = max(currSpeed, max(speeds))
            if maxSpeed == minSpeed:
                deltaDim = deltaDimLast
            else:
                deltaDim = int(((currSpeed - minSpeed) / (maxSpeed - minSpeed) + 0.5) * deltaDimLast)
            speeds.append(currSpeed)
        # update the current dimension
        currentDim = currentDim + deltaDim if currentDim + deltaDim <= dHigh else dHigh
        bsfTime = 0
        jumpTime = int((1 + (currentDim - dLow) / (dHigh - dLow)) * budget / beta)
        deltaDimLast = deltaDim
        print(f"!!! update the current dimension to {currentDim}")
        
        if deltaDim == 0:
            dims[currentDim].updateBounds()


# save optimizerNos to "seed_optNos.csv"
file_path = os.path.join(directory_path, f'seed_{seed}_optNos.csv')
with open(file_path, 'w') as file:
    for item in optimizerNos:
        file.write(str(item)+'\n')

# save qualities to "seed_qual.csv"
file_path = os.path.join(directory_path, f'seed_{seed}_qual.csv')
with open(file_path, 'w') as file:
    for item in dataset_Y.numpy():
        file.write(str(float(item))+'\n')

