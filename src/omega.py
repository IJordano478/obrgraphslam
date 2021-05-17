import numpy as np
import math
import logging

class Omega:
    # TODO: Confirm start location setup is correct e.g. if i start at (0,0,pi)
    def __init__(self, rss=3, lss=2):
        #Get robot and landmark state sizes
        self.rss = rss
        self.lss = lss
        self.omega_matrix = np.empty((rss, rss))       #transmatrix

        a = np.zeros((rss, rss))
        #np.fill_diagonal(a,np.inf)
        np.fill_diagonal(a, 0)
        self.omega_matrix[0:rss, 0:rss] = a

    def showOmegaOccupancy(self):
        size = ((self.omega_matrix.shape[0]-self.rss)//self.lss + 1, (self.omega_matrix.shape[0]-self.rss)//self.lss + 1)
        mat = np.zeros(size, dtype=int)
        if not np.allclose(self.omega_matrix[0:self.rss, 0:self.rss], np.zeros((self.rss, self.rss))):
            mat[0, 0] = 1

        for c in range(self.rss, self.omega_matrix.shape[0], self.lss):
            if not np.allclose(self.omega_matrix[0:self.rss, c:c+self.lss], np.zeros((self.rss, self.lss))):
                mat[0, (c - self.rss) // self.lss + 1] = 1
                mat[(c - self.rss) // self.lss + 1, 0] = 1

        for c in range(self.rss, self.omega_matrix.shape[0], self.lss):
            for r in range(self.rss, self.omega_matrix.shape[0], self.lss):
                if not np.allclose(self.omega_matrix[r:r+self.lss,c:c+self.lss], np.zeros((self.lss,self.lss))):
                    mat[(r - self.rss) // self.lss + 1, (c - self.rss) // self.lss + 1] = 1

        print(mat)
        return

    '''
    def showOmegaDetailed(self):
        for x in range(0, self.nPoses + self.nLmarks+1):
            for y in range(0, self.nPoses + self.nLmarks+1):
                print("Omega index: (", x, ",", y, ")\n", self.omega_matrix[x*3:(x+1)*3,y*3:(y+1)*3].round(3))
        return
    '''

class Xi:
    def __init__(self, rss=3):
        self.xi_vector = np.zeros((rss, 1)) #matrix where each row is an x,y,theta (poses) or x,y,signature (lmarks)