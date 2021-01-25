#!/usr/bin/env python
"""
GraphSLAM based on Thrun et al
"""
from src.frame2d import *
from src.omega import *

import numpy as np
import math
import copy

'''
our initial estimate will simply be provided by chaining together the motion model p(xt | ut, xt−1). Such an algorithm 
is outlined in Table 1, and called there GraphSLAM_initialize. This algorithm takes the controls u1:t as input, and 
outputs sequence of pose estimates µ0:t . It initializes the first pose by zero, and then calculates subsequent poses
by recursively applying the velocity motion model. Since we are only interested in the mean poses vector µ0:t , 
GraphSLAM_initialize only uses the deterministic part of the motion model. It also does not consider any measurement in 
its estimation.
'''
timeStep = 1
motionNoiseCovar = [1., 1., 1.]
measureNoiseCovar = [1., 1., 1.]

noiseCovarR = np.diag(motionNoiseCovar)
noiseCovarQ = np.diag(measureNoiseCovar)

landmarks = np.empty((0, 3))

#omega = None
#xi = None

def gs_initialise(controls: np.array):
    # Set base start pose and time step. Time step should be updated in integration version to actual time delta
    meanX = 0
    meanY = 0
    meanTheta = 0
    poses = np.zeros((1,3))

    # odometry motion model. Used w==0 to control singularity for straight line driving.
    i = 1
    for [v, w] in controls:
        if w == 0:
            meanX = meanX + (v * timeStep)
        else:
            meanX = meanX + ((-(v / w) * math.sin(meanTheta)) + ((v / w) * math.sin(meanTheta + w * timeStep)))
            meanY = meanY + (((v / w) * math.cos(meanTheta)) - ((v / w) * math.cos(meanTheta + w * timeStep)))
            meanTheta = meanTheta + (w * timeStep)

        #currPose = np.array([[math.cos(meanTheta), -math.sin(meanTheta), meanX], [math.sin(meanTheta), math.cos(meanTheta), meanY], [0., 0., 1.]])
        #poses = np.concatenate((poses,np.expand_dims(currPose, axis = 2)), axis=2)
        currPose = np.array([[meanX, meanY, meanTheta]])
        poses = np.concatenate((poses, currPose), axis=0)
        i += 1
    return poses


# 1: LINEARIZE(control, observations, correspondence, means)
def gs_linearize(controls=None, measurements=None, poseMeans=None, correspondence=None):
    # 2: Set omega and xi to 0
    #global omega, xi
    omega = Omega()
    xi = Xi()

    # 3: add 3x3 matrix with infinity on diagonals to omega at x0
    # Done in initialisation of Omega

    # 4: for all controls do
    for i in range(0,len(controls)):
        [v, w] = controls[i]

        # 5: xhat = the pose after the control has been applied
        xhat = poseMeans[i+1,:]

        # 6: G = 3x3 matrix of calculations
        jacobianG = np.eye(3)
        if w != 0:
            #jacobianG[0,2] = (-(v / w) * math.cos(angle2num(poseMeans[0:3,0:3,i]))) + ((v / w) * math.cos(angle2num(poseMeans[0:3,0:3,i]) + w * timeStep))
            #jacobianG[1,2] = (-(v / w) * math.cos(angle2num(poseMeans[0:3,0:3,i]))) + ((v / w) * math.cos(angle2num(poseMeans[0:3,0:3,i]) + w * timeStep))
            jacobianG[0,2] = (-(v / w) * math.cos(poseMeans[i,2])) + ((v / w) * math.cos(poseMeans[i,2] + w * timeStep))
            jacobianG[1,2] = (-(v / w) * math.cos(poseMeans[i,2])) + ((v / w) * math.cos(poseMeans[i,2] + w * timeStep))
        else:
            jacobianG[1, 2] = v*timeStep

        # 7: Add G and R to omega
        gt1 = np.vstack((np.transpose(-jacobianG), np.eye(3)))
        gt2 = np.hstack((-jacobianG, np.eye(3)))
        omegaUpdate = np.matmul(np.matmul(gt1, np.linalg.inv(noiseCovarR)), gt2)
        omega.addPose(omegaUpdate, i+1)
        #print("Omega:\n",omegaUpdate)

        # 8: add same with a bit more to xi
        #gt3 = np.hstack((xhat-jacobianG * poseMeans[0:3,0:3,i]))
        gt3 = xhat - np.matmul(jacobianG, poseMeans[i])
        xiUpdate = np.matmul(np.matmul(gt1, np.linalg.inv(noiseCovarR)), gt3)
        xi.addPose(xiUpdate, i+1)
        #print("Xi:\n", xiUpdate)

    # 9: endfor

    # 10: for all measurements zt do
    for i in range(0, measurements.shape[0]):
        measurement = measurements[i]
        time = int(measurement[0])
        print("Time:",time)
        # 11: Qt = sigma squared for r, phi and s
        # Skipped as same sensor used, so noise declared globally

        # 12: for all observed features at time
        #for m in measurements:

        # 13: j = observed landmark (c.i.t)
        # TODO j needs to be the actual landmark position that was noted, this is a placeholder that make ones from the
        #  measurement. j: [x,y,signature]
        pose = poseMeans[int(measurement[0]),:]
        pose2d = np.array([[1, 0, pose[0]],
                             [0, 1, pose[1]],
                             [0, 0, 1]])
        pose2d[0:2, 0:2] = num2matangle(pose[2])

        relative = np.array([[1, 0, math.cos(measurement[2])*measurement[1]],
                             [0, 1, math.sin(measurement[2])*measurement[1]],
                             [0, 0, 1]])
        relative[0:2,0:2] = num2matangle(measurement[2])
        lmark = np.matmul(pose2d, relative)
        j = lmark[0:3,2]
        j[2] = 0

        # TODO this part to next todo is poor, landmarks should be recorded during data collection
        match = False
        global landmarks
        for n in range(0,landmarks.shape[0]):
            if ((landmarks[n,0]-j[0])*(landmarks[n,0]-j[0]) + (landmarks[n,1]-j[1])*(landmarks[n,1]-j[1]) < (0.5)):
                j = landmarks[n,:]
                match = True
                index = n+1
                print("matched measurement to prexisting landmark")
                break

        if (match == False):
            print("no match from measurement, adding to landmarks")
            index = landmarks.shape[0] + 1
            if(landmarks.shape[0]==0):
                landmarks = np.expand_dims((j), axis=0)
            else:
                landmarks = np.concatenate((landmarks, np.expand_dims((j), axis=0)), axis=0)
        print("Landmark Index:", index)
        # TODO

        # 14: delta = [[deltax],[deltay]]
        delta = np.array([j[0]-pose[0], j[1]-pose[1]])

        # 15: q = transpose(delta*delta)
        q = np.matmul(np.transpose(delta),delta)

        # 16: zhat = (vector of stuff)
        zhat = np.array([math.sqrt(q), (math.atan2(delta[1],delta[0])-pose[2]), 0])

        # 17: H.i.t = Jacobian
        jacobianH = (1/q)*np.array([[-math.sqrt(q)*delta[0], -math.sqrt(q)*delta[1], 0, math.sqrt(q)*delta[0], math.sqrt(q)*delta[1], 0],
                                    [delta[1], -delta[0], -q, -delta[1], delta[0], 0],
                                    [0, 0, 0, 0, 0, q]])

        # 18: add H.i.t. and Qt^-1 to omega at xt and mj
        np.linalg.inv(noiseCovarQ)
        omegaUpdate = np.matmul(np.transpose(jacobianH), np.linalg.inv(noiseCovarQ))
        omegaUpdate = np.matmul(omegaUpdate, jacobianH)
        omega.addLandmark(omegaUpdate, time, index)

        # 19: add lots of stuff to xi
        ht2 = measurement[1:]-zhat + np.matmul(jacobianH, np.array([pose[0], pose[1], pose[2], j[0], j[1], 0]))
        xiUpdate = np.matmul(np.transpose(jacobianH), np.linalg.inv(noiseCovarQ))
        xiUpdate = np.matmul(xiUpdate, ht2)
        xi.addLandmark(xiUpdate, time, index)
        # 20: endfor

    # 21: endfor

    # 22: return omega, xi
    return omega, xi


# 1: REDUCE(omega, xi)
def gs_reduce(omega, xi):
    # 2: new omega = omega
    reducedOmega = copy.copy(omega)

    # 3: new xi = xi
    reducedXi = copy.copy(xi)

    print(xi.xiVector.round(4))
    # 4: for each feature j do
    for i in range(0, omega.nLmarks):

        # 5: let T(j) be the set of all poses xt that j was observed at
        # Ignored, attempting use of matrix to get all updates in one calculation
        # Note: update structure checks mathmatically, however actual update info may not be correct. Check with hand calculated

        # 6: Do some mathsy stuff to xi
        lmarkVisibility = omega.omegaMatrix[0:omega.nPoses+1,omega.nPoses+1+i,:,:] #The column of visibility from poses to the lmark
        mapDiagonal = np.expand_dims(omega.omegaMatrix[omega.nPoses+1+i,omega.nPoses+1+i], axis=0)
        updatePart1 = np.concatenate((lmarkVisibility,mapDiagonal), axis=0) #Adding the map element onto the bottom of the column

        mapDiagonalInv = np.linalg.pinv(mapDiagonal)

        updatePart1 = np.matmul(updatePart1, mapDiagonalInv)    #The lmark visibility column * inverse map diagonal element

        xiUpdatePart2 = np.identity(3)
        xiUpdatePart2[0:2,2] = np.array([xi.xiVector[omega.nPoses+1+i,0:2]])    #create a 3x3 from the map information given in xi
        xiUpdate = np.matmul(updatePart1,xiUpdatePart2)         #complete the xi update equation

        #convert 3x3 mat to 1x3 vector for xi
        xiUpdateVector = np.zeros((xiUpdate.shape[0]-1,3))
        for n in range(0, omega.nPoses+1):
            xiUpdateVector[n,:] = np.array([xiUpdate[n,0,2],xiUpdate[n,1,2], angle2num(xiUpdate[n,:,:])])

        #update poses and the map value
        reducedXi.xiVector[0:omega.nPoses+1,:] -= xiUpdateVector
        reducedXi.xiVector[omega.nPoses+1+i,:] -= np.array([xiUpdate[-1,0,2],xiUpdate[-1,1,2], angle2num(xiUpdate[-1,:,:])])


        # 7: Do some mathsy stuff to omega
        updatePart1 = np.expand_dims((updatePart1),axis=1) #correct dimensionality to 4d column
        omegaUpdatePart2 = omega.omegaMatrix[omega.nPoses+1+i, 0:omega.nPoses+1,:,:]   #get the row of visibility from poses to lmark
        omegaUpdatePart2 = np.expand_dims((omegaUpdatePart2),axis=0) #correct dimensionality to 4d row
        omegaUpdatePart2 = np.concatenate((omegaUpdatePart2, np.expand_dims((mapDiagonal),axis=0)), axis=1)      #add onto the end the map diagonal
        omegaUpdate = np.matmul(updatePart1, omegaUpdatePart2)                          #complete the omega update equation
        #print("omegaUpdate\n",omegaUpdate)

        #Apply update to omega at square of poses, column observation, row observation and map diagonal
        reducedOmega.omegaMatrix[0:omega.nPoses+1,0:omega.nPoses+1,:,:] -= omegaUpdate[0:-1,0:-1,:,:]
        reducedOmega.omegaMatrix[0:omega.nPoses+1,omega.nPoses+1+i,:,:] -= omegaUpdate[0:-1,-1,:,:]
        reducedOmega.omegaMatrix[omega.nPoses+1+i,0:omega.nPoses+1,:,:] -= omegaUpdate[-1,0:-1,:,:]
        reducedOmega.omegaMatrix[omega.nPoses+1+i,omega.nPoses+1+i,:,:] -= omegaUpdate[-1,-1,:,:]

    # 8: Remove from omega and xi all rows and columns corresponding to j
    # Performed as block after to prevent index changing during iterations
    reducedXi.xiVector = reducedXi.xiVector[0:omega.nPoses+1,:]
    reducedXi.nPoses = reducedXi.xiVector.shape[0]-1
    reducedXi.nLmarks = 0

    reducedOmega.omegaMatrix = reducedOmega.omegaMatrix[0:omega.nPoses+1,0:omega.nPoses+1,:,:]
    reducedOmega.nPoses = reducedXi.xiVector.shape[0] - 1
    reducedOmega.nLmarks = 0

    # 9: endfor

    # 10: return new omega, new xi
    return reducedOmega, reducedXi


# 1: SOLVE(newOmega,newXi,Omega,Xi)
def gs_solve():
    # 2: SumSigma = newOmega inverse

    # 3: means = sumSigma * newXi

    # 4: for each feature j do

    # 5: set T(j) to the set of all poses xt that j was observed at

    # 6: meanJ = mathsy stuff that i won't write now

    # 7: endfor

    # 8: return mean, sumSigma
    return


# 1: TEST(omega, xi, mean, landmark1, landmark2)
def gs_known_correspondence_test(omega, xi, poseMean, pathCovariance, lmarkJ, lmarkK):
    # 2-7: A butt tonne of maths



    return


def GraphSLAM():
    return


def angle2num(matAngle):
    return math.atan2(matAngle[1, 0], matAngle[0, 0])

def num2matangle(angle):
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])

if __name__ == "__main__":
    # controls = np.array([[1,0],[1,0],[0,math.pi/2],[1,0],[1,0],[0,math.pi/2],[1,0]])
    # controls = np.array([[2, math.pi/2], [2, math.pi/2], [2, math.pi/2], [2, math.pi/2]])

    pose1 = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
    pose2 = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 1]])
    pose3 = np.array([[0, -1, 3], [1, 0, 1], [0, 0, 1]])
    measurements = np.array([[1, math.sqrt(2), math.pi/4, 0],
                             [1, math.sqrt(5), -0.463647, 0],
                             [2, math.sqrt(2), -math.pi/4, 0],
                             [3, 1, 0, 0]])
    controls = np.array([[1, 0], [1, 0], [math.pi / 2, math.pi / 2]])

    #omega.addPose(pose1)
    #omega.addPose(pose2)

    #controls = np.array(
    #    [[1, 0], [1, 0], [2, math.pi / 2], [2, math.pi / 2], [2, math.pi / 2], [2, math.pi / 2], [0, math.pi], [2, 0],
    #     [0, math.pi]])

    #measurements = None
    meanPoses = gs_initialise(controls)

    print("Recorded poses:")
    for i in range(0, meanPoses.shape[0]):
        print(np.round(meanPoses[i,:], 3))
    omega, xi = gs_linearize(controls,measurements, meanPoses)
    omega.showOmega()
    print("\n")
    #print(xi.xiVector)
    gs_reduce(omega,xi)
