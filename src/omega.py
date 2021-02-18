import numpy as np
import math
import logging

class Omega:
    def __init__(self):
        self.nPoses = 0
        self.nLmarks = 0
        self.poses = np.empty((3, 3, 0))                #3x3 x depth
        self.landmarks = np.empty((3, 3, 0))            #3x3 x depth
        self.omegaMatrix = np.empty((1, 1, 3, 3))       #row x col x transmatrix

        self.poses = np.concatenate((self.poses, np.expand_dims(np.identity(3), axis=2)), axis=2)

        a = np.zeros((3, 3))
        #np.fill_diagonal(a,np.inf)
        np.fill_diagonal(a, 100)
        self.omegaMatrix[0,0] = a

    def addPose(self, newPose, time):
        #If a pose is already recorded for this time
        if(time <= self.nPoses):
            print("Something has gone wrong here")

        # else the time is beyond currently stored poses, a resize is needed
        else:

            xx = self.omegaMatrix[0:self.nPoses + 1, 0:self.nPoses + 1, :, :]

            # if landmarks exist, adapt size of xx, xm and mx by adding 1 to the relative dimension filled with
            # 3x3 matices containing zeros
            if(self.nLmarks > 0):
                '''
                newCol = np.zeros((self.nPoses + 1, 1, 3, 3))
                newRow = np.zeros((1, self.nPoses + 2, 3, 3))
                xx = np.concatenate((xx, newCol), axis=1)
                xx = np.concatenate((xx, newRow), axis=0)

                newRow = np.zeros((1, self.nLmarks, 3, 3))
                xm = self.omegaMatrix[0:self.nPoses + 1, self.nPoses + self.nLmarks + 1, :, :]
                xm = np.concatenate((xm, newRow), axis=0)

                newCol = np.zeros((self.nLmarks, 1, 3, 3))
                mx = self.omegaMatrix[self.nPoses+self.nLmarks+1, 0:self.nPoses+1, :, :]
                mx = np.concatenate((mx, newCol), axis=1)

                mm = self.omegaMatrix[self.nPoses+self.nLmarks+1, self.nPoses+self.nLmarks+1, :, :]

                topOmega = np.concatenate((xx, xm), axis=1)
                bottomOmega = np.concatenate((mx, mm), axis=1)

                self.omegaMatrix = np.concatenate((topOmega, bottomOmega), axis=0)
                '''
            # No landmarks currently stored, just need to adapt the xx portion of the matrix
            else:
                newCol = np.zeros((self.nPoses+1,1,3,3))
                newRow = np.zeros((1, self.nPoses + 2, 3, 3))
                xx = np.concatenate((xx, newCol), axis=1)
                xx = np.concatenate((xx, newRow), axis=0)
                self.omegaMatrix = xx

            #Use update to adjust relevant entries
            self.omegaMatrix[time-1, time-1, :, :] += newPose[0:3, 0:3]
            self.omegaMatrix[time-1, time, :, : ] += newPose[0:3, 3:6]
            self.omegaMatrix[time, time-1, :, :] += newPose[3:6, 0:3]
            self.omegaMatrix[time, time, :, :] += newPose[3:6, 3:6]

            #print("Omega Matrix shape:", self.omegaMatrix.shape)

        self.nPoses += 1
        return

    def addLandmark(self, newLandmark, time, lmarkIndex):
        if (lmarkIndex > self.nLmarks):
            #corresponds to new landmark
            newCol = np.zeros((self.nPoses + self.nLmarks + 1, 1, 3, 3))
            newRow = np.zeros((1, self.nPoses + self.nLmarks + 2, 3, 3))
            self.omegaMatrix = np.concatenate((self.omegaMatrix, newCol), axis=1)
            self.omegaMatrix = np.concatenate((self.omegaMatrix, newRow), axis=0)

            self.omegaMatrix[time, time, :, :] += newLandmark[0:3, 0:3]
            self.omegaMatrix[time, -1, :, :] += newLandmark[0:3, 3:6]
            self.omegaMatrix[-1, time, :, :] += newLandmark[3:6, 0:3]
            self.omegaMatrix[-1, -1, :, :] += newLandmark[3:6, 3:6]
            #print("Omega Matrix shape:", self.omegaMatrix.shape)

            self.nLmarks += 1
        else:
            #corresponds to prexisting landmark
            self.omegaMatrix[time, time, :, :] += newLandmark[0:3, 0:3]
            self.omegaMatrix[time, self.nPoses+lmarkIndex, :, :] += newLandmark[0:3, 3:6]
            self.omegaMatrix[self.nPoses+lmarkIndex, time, :, :] += newLandmark[3:6, 0:3]
            self.omegaMatrix[self.nPoses+lmarkIndex, self.nPoses+lmarkIndex, :, :] += newLandmark[3:6, 3:6]

        return

    def updatePose(self, pos, omegaUpdate):
        self.omegaMatrix[pos[0]][pos[1]][:][:] = omegaUpdate
        self.omegaMatrix[pos[1]][pos[2]][:][:] = omegaUpdate
        return

    def showOmegaOccupancy(self):
        mat = np.zeros((self.nPoses + self.nLmarks + 1, self.nPoses + self.nLmarks + 1), dtype=int)
        for x in range(0, self.nPoses + self.nLmarks+1):
            for y in range(0, self.nPoses + self.nLmarks+1):
                if np.allclose(self.omegaMatrix[x,y,:,:], np.zeros(3)):
                    mat[x,y] = 0
                else:
                    mat[x,y] = 1

        print(mat)
        return

    def showOmegaDetailed(self):
        for x in range(0, self.nPoses + self.nLmarks+1):
            for y in range(0, self.nPoses + self.nLmarks+1):
                print("Omega index: (", x, ",", y, ")\n", self.omegaMatrix[x, y,:,:].round(3))
        return

class Omega2:
    def __init__(self):
        self.nPoses = 0
        self.nLmarks = 0
        self.omegaMatrix = np.empty((3, 3))       #transmatrix

        a = np.zeros((3, 3))
        #np.fill_diagonal(a,np.inf)
        np.fill_diagonal(a, 1)
        self.omegaMatrix[0:3, 0:3] = a

    def addPose(self, newPose, time):

        #If a pose is already recorded for this time
        if(time <= self.nPoses):
            print("Something has gone wrong here")

        # else the time is beyond currently stored poses, a resize is needed
        else:

            xx = self.omegaMatrix[0:(self.nPoses + 1)*3, 0:(self.nPoses + 1)*3]

            # if landmarks exist, adapt size of xx, xm and mx by adding 1 to the relative dimension filled with
            # 3x3 matices containing zeros
            if(self.nLmarks > 0):
                return
            # No landmarks currently stored, just need to adapt the xx portion of the matrix
            else:
                newCol = np.zeros(((self.nPoses+1)*3,3))
                newRow = np.zeros((3, (self.nPoses + 2)*3))
                xx = np.concatenate((xx, newCol), axis=1)
                xx = np.concatenate((xx, newRow), axis=0)
                self.omegaMatrix = xx

            #Use update to adjust relevant entries
            self.omegaMatrix[(time-1)*3:(time)*3, (time-1)*3:(time)*3] += newPose[0:3, 0:3]
            self.omegaMatrix[(time-1)*3:(time)*3, (time)*3:(time+1)*3] += newPose[0:3, 3:6]
            self.omegaMatrix[(time)*3:(time+1)*3, (time-1)*3:(time)*3] += newPose[3:6, 0:3]
            self.omegaMatrix[(time)*3:(time+1)*3, (time)*3:(time+1)*3] += newPose[3:6, 3:6]

        self.nPoses += 1
        return

    def addLandmark(self, newLandmark, time, lmarkIndex):
        if (lmarkIndex > self.nLmarks):
            #corresponds to new landmark
            newCol = np.zeros(((self.nPoses + self.nLmarks + 1)*3, 3))
            newRow = np.zeros((3, (self.nPoses + self.nLmarks + 2)*3))
            self.omegaMatrix = np.concatenate((self.omegaMatrix, newCol), axis=1)
            self.omegaMatrix = np.concatenate((self.omegaMatrix, newRow), axis=0)

            self.omegaMatrix[(time)*3:(time+1)*3, (time)*3:(time+1)*3] += newLandmark[0:3, 0:3]
            self.omegaMatrix[(time)*3:(time+1)*3, -4:-1] += newLandmark[0:3, 3:6]
            self.omegaMatrix[-4:-1, (time)*3:(time+1)*3] += newLandmark[3:6, 0:3]
            self.omegaMatrix[-4:-1, -4:-1] += newLandmark[3:6, 3:6]
            #print("Omega Matrix shape:", self.omegaMatrix.shape)

            self.nLmarks += 1
        else:
            #corresponds to prexisting landmark
            self.omegaMatrix[(time)*3:(time+1)*3, (time)*3:(time+1)*3] += newLandmark[0:3, 0:3]
            self.omegaMatrix[(time)*3:(time+1)*3, (self.nPoses+lmarkIndex)*3:(self.nPoses+lmarkIndex+1)*3] += newLandmark[0:3, 3:6]
            self.omegaMatrix[(self.nPoses+lmarkIndex)*3:(self.nPoses+lmarkIndex+1)*3, (time)*3:(time+1)*3] += newLandmark[3:6, 0:3]
            self.omegaMatrix[(self.nPoses+lmarkIndex)*3:(self.nPoses+lmarkIndex+1)*3, (self.nPoses+lmarkIndex)*3:(self.nPoses+lmarkIndex+1)*3] += newLandmark[3:6, 3:6]

        return

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

    def addPose(self, pos, time):

        if(time <= self.nPoses):
            print("Something has gone horribly wrong")
        else:
            self.xiVector = np.concatenate((self.xiVector, np.zeros((3,3))), axis=0)
            self.xiVector[(time-1)*3:(time)*3,:] += pos[0:3,:]
            self.xiVector[(time)*3:(time+1)*3,:] += pos[3:6,:]
            #self.xiVector[time, :] += pos[3:6] + self.xiVector[time-1,:]
            self.nPoses+=1
        return

    def addLandmark(self, update, time, lmarkIndex):
        if(lmarkIndex > self.nLmarks):
            # Corresponds to new landmark
            self.xiVector = np.concatenate((self.xiVector, np.zeros((1, 3))), axis=0)
            self.xiVector[time, :] += update[0:3]
            self.xiVector[self.nPoses + lmarkIndex,:] += update[3:6]
            self.nLmarks += 1

            #print("xi shape", self.xiVector.shape)
            #print(self.xiVector)
        else:
            # Corresponds to prexisting landmark
            self.xiVector[time, :] += update[0:3]
            self.xiVector[self.nPoses+lmarkIndex, :] += update[3:6]
            #print("xi shape", self.xiVector.shape)
            #print(self.xiVector)

        return