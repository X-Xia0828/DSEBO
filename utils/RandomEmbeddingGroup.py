import torch
import numpy as np

torch.set_default_dtype(torch.double)


class RandomEmbeddingGroup():
    def __init__(self, objDim, maxOptDim, objBounds=None, seed=0, dist='gaussian'):
        self.objDim = objDim
        self.maxOptDim = maxOptDim
        self.seed = seed
        
        if objBounds == None: 
            self.objBounds = torch.stack([torch.ones(objDim) * (-1), torch.ones(objDim)])
        else:
            self.objBounds = objBounds
        
        torch.manual_seed(seed)
        
        if dist == 'gaussian':
            self.randomMatrix = torch.randn(objDim, maxOptDim) / np.sqrt(maxOptDim) 
        elif dist == 'uniform':
            self.randomMatrix = torch.rand(objDim, maxOptDim) / np.sqrt(maxOptDim) 
        else:
            raise ValueError('Invalid distribution type')
        self.dist = dist
        
    def __call__(self, dim, x):
        if dim > self.maxOptDim:
            raise ValueError('Invalid dimension')
        embeddingMat = self.randomMatrix[:, :dim]
        obj_x = torch.matmul(embeddingMat, x.t()).t()
        batches = obj_x.size()[0]
        clipped_x = torch.stack([torch.max(obj_x[i], self.objBounds[0]) for i in range(batches)])
        clipped_x = torch.stack([torch.min(clipped_x[i], self.objBounds[1]) for i in range(batches)])
        return clipped_x
    
    def dimProjection(self, dimLow, dimHigh, x):
        if dimHigh > self.maxOptDim or dimLow >= dimHigh:
            raise ValueError('Invalid dimension')
        matLow = self.randomMatrix[:, :dimLow]
        matHighInv = torch.pinverse(self.randomMatrix[:, :dimHigh])
        obj_x = torch.matmul(matHighInv, torch.matmul(matLow, x.t())).t()
        return obj_x
    
    def getFullRandomMatrix(self):
        return self.randomMatrix
    
    def getDistribution(self):
        return self.dist
    
    def getObjDim(self):
        return self.objDim
    
    def getMaxOptDim(self):
        return self.maxOptDim
    
    def getSeed(self):
        return self.seed 


