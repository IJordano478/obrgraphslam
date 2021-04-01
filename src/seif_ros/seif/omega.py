import numpy as np
import math
import logging


class Omega2:
    def __init__(self):
        self.nPoses = 0
        self.nLmarks = 0
        self.omegaMatrix = np.empty((3, 3))       #transmatrix

        a = np.zeros((3, 3))
        #np.fill_diagonal(a,np.inf)
        np.fill_diagonal(a, 1)
        self.omegaMatrix[0:3, 0:3] = a

    def showOmegaOccupancy(self):
        size = ((self.omegaMatrix.shape[0]-3)//2 + 1, (self.omegaMatrix.shape[0]-3)//2 + 1)
        mat = np.zeros(size, dtype=int)
        if not np.allclose(self.omegaMatrix[0:3, 0:3], np.zeros((3, 3))):
            mat[0, 0] = 1

        for c in range(3, self.omegaMatrix.shape[0], 2):
            if not np.allclose(self.omegaMatrix[0:3, c:c+2], np.zeros((3, 2))):
                mat[0, (c - 3) // 2 + 1] = 1
                mat[(c - 3) // 2 + 1, 0] = 1

        for c in range(3, self.omegaMatrix.shape[0], 2):
            for r in range(3, self.omegaMatrix.shape[0], 2):
                if not np.allclose(self.omegaMatrix[r:r+2,c:c+2], np.zeros((2,2))):
                    mat[(r - 3) // 2 + 1, (c - 3) // 2 + 1] = 1

        print(mat)
        return

    def showOmegaDetailed(self):
        for x in range(0, self.nPoses + self.nLmarks+1):
            for y in range(0, self.nPoses + self.nLmarks+1):
                print("Omega index: (", x, ",", y, ")\n", self.omegaMatrix[x*3:(x+1)*3,y*3:(y+1)*3].round(3))
        return


class Xi:
    def __init__(self):
        self.nPoses = 0
        self.nLmarks = 0
        self.xiVector = np.zeros((3, 1)) #matrix where each row is an x,y,theta (poses) or x,y,signature (lmarks)

