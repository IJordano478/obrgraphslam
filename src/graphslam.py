#!/usr/bin/env python
"""
GraphSLAM based on Thrun et al
"""
from src.frame2d import *

import numpy as np
import math

'''
our initial estimate will simply be provided by chaining together the motion model p(xt | ut, xt−1). Such an algorithm 
is outlined in Table 1, and called there GraphSLAM_initialize. This algorithm takes the controls u1:t as input, and 
outputs sequence of pose estimates µ0:t . It initializes the first pose by zero, and then calculates subsequent poses
by recursively applying the velocity motion model. Since we are only interested in the mean poses vector µ0:t , 
GraphSLAM_initialize only uses the deterministic part of the motion model. It also does not consider any measurement in 
its estimation.
'''


def gs_initialise(controls: np.array):
    # Set base start pose and time step. Time step should be updated in integration version to actual time delta
    timeStep = 1
    meanX = 0
    meanY = 0
    meanTheta = 0

    poses = np.empty(len(controls) + 1, dtype=Frame2D)
    poses[0] = Frame2D().fromXYA(meanX, meanY, meanTheta)

    # odometry motion model. Used w==0 to control singularity for straight line driving.
    i = 1
    for [v, w] in controls:
        if w == 0:
            meanX = meanX + (v * math.cos(meanTheta)) * timeStep
            meanY = meanY + (v * math.sin(meanTheta)) * timeStep
        else:
            meanX = meanX + ((-(v / w) * math.sin(meanTheta)) + ((v / w) * math.sin(meanTheta + w * timeStep)))
            meanY = meanY + (((v / w) * math.cos(meanTheta)) - ((v / w) * math.cos(meanTheta + w * timeStep)))
            meanTheta = meanTheta + (w * timeStep)
        poses[i] = Frame2D().fromXYA(meanX, meanY, meanTheta)
        i += 1

    return poses


# 1: LINEARIZE(control, observations, correspondence, means)
def gs_linearize(controls=None, measurements=None, poseMeans=None, correspondence=None):
    # 2: Set omega and xi to 0
    omega = np.mat([1,1], dtype=Frame2D)
    xi = 0

    # 3: add 3x3 matrix with infinity on diagonals to omega at x0
    infMat = np.zeros((3, 3))
    np.fill_diagonal(infMat, np.inf)
    omega[0, 0] = Frame2D().fromMat(infMat)

    # 4: for all controls do
    for i in range(0,len(controls)):
        [v, w] = controls[i]

        # 5: xhat = calculations
        xhat = poseMeans[i+1]

        # 6: G = 3x3 matrix of calculations
        jacobianG = Frame2D()
        jacobianG.mat[0:2,2] = poseMeans[i+1].mat[0:2,2]

        # 7: Add G and R to omega
        motionNoiseCovar = [0.01, 0.01, 0.01]
        R = np.diag(motionNoiseCovar)
        R = Frame2D().fromMat(R)
        omegaUpdate = np.vstack((np.transpose(-jacobianG.mat), [1, 1, 1])) * \
            R.inverse().mat * np.hstack((-jacobianG.mat, np.array([[1], [1], [1]])))
        #TODO use update

        # 8: add same with a bit more to xi
        xiUpdate = np.vstack((np.transpose(-jacobianG.mat), [1, 1, 1])) * R.inverse().mat * np.hstack((xhat-jacobianG, poseMeans[i]))
        #TODO use update

    # 9: endfor

    # 10: for all measurements zt do
    for i in range(0, len(measurements)):
        # 11: Qt = sigma squared for r, phi and s
        measureNoiseCovar = [0.01, 0.01, 0.01]
        Q = np.diag(measureNoiseCovar)
        Q = Frame2D().fromMat(Q)

        # 12: for all observed features

            # 13: j = observed landmark (c.i.t)

            # 14: delta = [[deltax],[deltay]]

            # 15: q = transpose(delta*delta)

            # 16: zhat = (vector of stuff)

            # 17: H.i.t = Jacobian

            # 18: add H.i.t. and Qt^-1 to omega at xt and mj

            # 19: add lots of stuff to xi

        # 20: endfor

    # 21: endfor

    # 22: return omega, xi
    return


# 1: REDUCE(omega, xi)
def gs_reduce(omega, xi):
    # 2: new omega = omega
    reducedOmega = omega

    # 3: new xi = xi
    reducedXi = xi

    # 4: for each feature j do


    # 5: let T(j) be the set of all poses xt that j was observed at

    # 6: Do some mathsy stuff to xi

    # 7: Do some mathsy stuff to omega

    # 8: Remove from omega and xi all rows and columns corresponding to j

    # 9: endfor

    # 10: return new omega, new xi
    return


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


# 1: TEST(omega, xi, mean, )
def gs_known_correspondence_test():
    # 2-7: A butt tonne of maths
    return


def GraphSLAM():
    return


if __name__ == "__main__":
    # controls = np.array([[1,0],[1,0],[0,math.pi/2],[1,0],[1,0],[0,math.pi/2],[1,0]])
    # controls = np.array([[2, math.pi/2], [2, math.pi/2], [2, math.pi/2], [2, math.pi/2]])
    controls = np.array(
        [[1, 0], [1, 0], [2, math.pi / 2], [2, math.pi / 2], [2, math.pi / 2], [2, math.pi / 2], [0, math.pi], [2, 0],
         [0, math.pi]])
    measurements = None

    meanPoses = gs_initialise(controls)
    for pos in meanPoses:
        print(pos)
    gs_linearize(controls,measurements, meanPoses)
