#!/usr/bin/env python
"""
GraphSLAM based on Thrun et al


"Sparse Bayesian Information Filters for Localization and Mapping" by Matthew Walter)
issue with overconfidence
"""
from src.frame2d import *
from src.omega import *
from scipy.linalg import fractional_matrix_power
from scipy.stats import multivariate_normal

import numpy as np
import math
import copy
from collections import deque

timeStep = 1
motionNoiseCovariance = [1.**2, 1.**2, 1.**2]
measureNoiseCovariance = [1.**2, 1.**2]

noiseCovarianceR = np.diag(motionNoiseCovariance)
noiseCovarianceQ = np.diag(measureNoiseCovariance)

landmarks = np.empty((0, 0))
sparsityN = 0
active = 0
toDeactivate = 0


def seif_known_correspondence(xi: Xi, omega: Omega2, mean, newControl, newMeasurements):
    xi, omega, mean = seif_motion_update(xi, omega, mean, newControl)
    xi, omega = seif_measurement_update(xi, omega, mean, newMeasurements)
    mean = seif_update_state_estimation(xi, omega, mean)
    xi, omega - seif_sparsification(xi, omega)
    return xi, omega, mean


def seif_motion_update(xi, omega, mean, control):
    # breakpoint()
    print("mean at time t-1:\n",mean.round(3))
    print("control (", control[0], ",", control[1], ")")
    v, w = control

    # 2
    Fx = np.concatenate((np.eye(3), np.zeros((3, mean.shape[0]-3))), axis=1)
    FxTranspose = np.transpose(Fx)

    # 3
    w += 0.0000001  # Singularity avoidance
    delta = np.array([[-(v / w) * math.sin(mean[2, 0]) + (v / w) * math.sin(mean[2, 0] + w * timeStep)],
                      [(v / w) * math.cos(mean[2, 0]) - (v / w) * math.cos(mean[2, 0] + w * timeStep)],
                      [w * timeStep]])

    # 4 #ERRATA says this should be negative
    deltaMat = np.array([[0, 0, (v / w) * math.cos(mean[2, 0]) - (v / w) * math.cos(mean[2, 0] + w * timeStep)],
                         [0, 0, (v / w) * math.sin(mean[2, 0]) - (v / w) * math.sin(mean[2, 0] + w * timeStep)],
                         [0, 0, 0]])

    # 5
    psi = FxTranspose @ deltaMat @ Fx
    psiTranspose = np.transpose(psi)

    # 6
    lambda_ = (psiTranspose @ omega.omegaMatrix) + (omega.omegaMatrix @ psi) + (psiTranspose @ omega.omegaMatrix @ psi)

    # 7
    phi = omega.omegaMatrix + lambda_

    # 8
    kappa = phi @ FxTranspose @ (np.linalg.inv(np.linalg.inv(noiseCovarianceR) + Fx @ phi @ FxTranspose)) @ Fx @ phi

    # 9
    omega.omegaMatrix = phi - kappa

    # 10
    xi.xiVector = xi.xiVector + ((lambda_ - kappa) @ mean) + (omega.omegaMatrix @ FxTranspose @ delta)

    # 11
    mean[0:3,:] += delta

    print("mean at time t:\n", mean.round(3))
    # 12
    # breakpoint()
    return xi, omega, mean


def seif_measurement_update(xi, omega, mean, measurements):
    # breakpoint()
    toDeactivate = deque(maxlen=sparsityN)

    # 3
    xiSumMeasurements = np.zeros((1, 1))
    omegaSumMeasurements = np.zeros((1, 1))
    for range_, bearing, signature in measurements:
        z = np.array([[range_], [bearing]])

        # 4,5,6
        # Take measurement and check it against existing landmarks. If its within a certain distance then assume same
        # landmark
        mu_j = mean[0:2, :] + range_ * np.array([[math.cos(bearing + mean[2, 0])], [math.sin(bearing + mean[2, 0])]])

        # This will likely become a correspondence test
        found = False
        i = 0
        # for x, y in landmarks:
        for n in range(0, (mean.shape[0]-3)//2):
            # TODO include signature match
            # if (mu_j[0, 0] - landmarks[n,0])**2 + (mu_j[1, 0] - landmarks[n+1,0])**2 < 1:
            if (mu_j[0, 0] - mean[3+2*n, 0]) ** 2 + (mu_j[1, 0] - mean[4+2*n, 0]) ** 2 < 1:
                mu_j = mean[3+2*n:3+2*n+2, :]
                found = True
                break
            i += 1

        # landmark not seen before, add to map
        if not found:
            mean = np.concatenate((mean, mu_j), axis=0)

        # manage the active landmarks and mark those that need to be deactivated on this timestep
        if i not in active:
            if len(active) == sparsityN and len(toDeactivate) != sparsityN:
                toDeactivate = active[0]
            active.append(i)
        print("active indexes =", active)

        # 8
        delta = mu_j - mean[0:2, :]

        # 9
        q = (np.transpose(delta) @ delta)[0][0]

        # 10
        zhat = np.array([[math.sqrt(q)],
                         [math.atan2(delta[1, 0], delta[0,0]) - mean[2, 0]]])

        #print(mean.round(3))
        # 11
        # jacobian H must be made of 4 different matrix components; the pose, 0s, the landmark, 0s.
        h1 = np.array([[-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0], 0],
                       [delta[1, 0], -delta[0, 0], -q]])

        h2 = np.zeros((2, 2 * (i+1) - 2))

        h3 = np.array([[math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0]],
                       [-delta[1, 0], delta[0, 0]]])

        h4 = np.zeros((2, (mean.shape[0]-3)//2 - (i+1)))

        h_it = 1 / q * np.concatenate((h1, h2, h3, h4), axis=1)

        #meanXiUpdate = np.zeros((h_it.shape[1], 1))
        #meanXiUpdate[0:3, :] = mean[0:3,:]
        #meanXiUpdate[3+(2*i):3+(2*i)+2, :] = mu_j

        # xiUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat +
        # breakpoint()
        #xiUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat + (h_it @ meanXiUpdate))
        xiUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat + (h_it @ mean))
        # resize array with new columns and rows as zeros
        if xiUpdate.shape > xiSumMeasurements.shape:
            newArray = np.zeros(xiUpdate.shape)
            newArray[0:xiSumMeasurements.shape[0], 0:xiSumMeasurements.shape[1]] = xiSumMeasurements
            xiSumMeasurements = newArray

        omegaUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ h_it
        if omegaUpdate.shape > omegaSumMeasurements.shape:
            newMatrix = np.zeros(omegaUpdate.shape)
            newMatrix[0:omegaSumMeasurements.shape[0], 0:omegaSumMeasurements.shape[0]] = omegaSumMeasurements
            omegaSumMeasurements = newMatrix

        # xiSumMeasurements += np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat + (h_it @ mean[0:2]))
        # xiSumMeasurements += np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat + np.multiply(h_it, mean[0:2]))
        xiSumMeasurements += xiUpdate
        omegaSumMeasurements += omegaUpdate

    # 13, 14
    if xiSumMeasurements.shape > xi.xiVector.shape:
        newArray = np.zeros(xiSumMeasurements.shape)
        newArray[0:xi.xiVector.shape[0], 0:xi.xiVector.shape[1]] = xi.xiVector
        xi.xiVector = newArray

        newMatrix = np.zeros(omegaSumMeasurements.shape)
        newMatrix[0:omega.omegaMatrix.shape[0], 0:omega.omegaMatrix.shape[0]] = omega.omegaMatrix
        omega.omegaMatrix = newMatrix

    xi.xiVector += xi.xiVector + xiSumMeasurements
    omega.omegaMatrix += omega.omegaMatrix + omegaSumMeasurements

    # 15
    #print((np.linalg.inv(omega.omegaMatrix) @ xi.xiVector).round(3))
    omega.nPoses = (omega.omegaMatrix.shape[0]-3)//2
    # breakpoint()
    return xi, omega, mean


def seif_update_state_estimation(xi, omega, mean):
    # breakpoint()
    global active
    #mean = np.concatenate((mean, landmarks), axis=0)

    # 2
    #for small number of active map features
    for i in active:
        Fi1 = np.zeros((2, 3+2*i))
        Fi2 = np.identity(2)
        Fi3 = np.zeros((2, (mean.shape[0]-3) - 2*(i+1)))
        Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)
        FiTranspose = np.transpose(Fi)

        mean_it = np.linalg.inv(Fi @ omega.omegaMatrix @ FiTranspose) @ Fi @ (xi.xiVector - (omega.omegaMatrix @ mean) + omega.omegaMatrix @ FiTranspose @ Fi @ mean)
        mean[3+(i*2):3+(i*2)+2, :] = mean_it

    # 9
    Fx = np.concatenate((np.identity(3), np.zeros((3, mean.shape[0]-3))), axis=1)
    FxTranspose = np.transpose(Fx)

    # 10
    updatedMean = np.linalg.inv(Fx @ omega.omegaMatrix @ FxTranspose) @ Fx @ (xi.xiVector - (omega.omegaMatrix @ mean) + (omega.omegaMatrix @ FxTranspose @ Fx @ mean))
    mean[0:3,:] = updatedMean
    # 11
    # breakpoint()
    return mean


def seif_sparsification(xi, omega, mean):
    # breakpoint()
    global active
    global toDeactivate

    # 2 Generate projection matrices
    omegaSize = omega.omegaMatrix.shape

    F_xActDeact = np.zeros((omegaSize))
    F_xActDeact[0:3,0:3] = np.identity(3)

    F_deact = np.zeros((omegaSize))

    F_xDeact = np.zeros((omegaSize))
    F_xDeact[0:3, 0:3] = np.identity(3)

    F_x = np.zeros((omegaSize))
    F_x[0:3,0:3] = np.identity(3)

    for i in active:
        F_xActDeact[3+(i*2):5+(i*2), 3+(i*2):5+(i*2)] = np.identity(2)

    for i in toDeactivate:
        F_xActDeact[3 + (i*2):5 + (i*2), 3 + (i*2):5 + (i*2)] = np.identity(2)
        F_xDeact[3 + (i*2):5 + (i*2), 3 + (i*2):5 + (i*2)] = np.identity(2)
        F_deact[3 + (i*2):5 + (i*2), 3 + (i*2):5 + (i*2)] = np.identity(2)

    # 2.1 Missing from book
    omega_0t = F_xActDeact @ omega.omegaMatrix @ F_xActDeact

    omega1 = -(omega_0t @ F_deact @ np.linalg.pinv(F_deact @ omega_0t @ F_deact) @ F_deact @ omega_0t)

    omega2 = omega_0t @ F_xDeact @ np.linalg.pinv(F_xDeact @ omega_0t @ F_xDeact) @ F_xDeact @ omega_0t

    omega3 = -(omega.omegaMatrix @ F_x @ np.linalg.pinv(F_x @ omega.omegaMatrix @ F_x) @ F_x @ omega.omegaMatrix)

    sparsifiedOmega = omega.omegaMatrix + omega1 + omega2 + omega3

    sparsifiedXi = xi.xiVector + (sparsifiedOmega - omega.omegaMatrix) @ mean

    #print((np.linalg.inv(sparsifiedOmega) @ sparsifiedXi).round(3))
    # breakpoint()
    return xi, omega


def seif_correspondence_test():
    return


# def xya_to_matrix(xya):
#    if len(xya.shape) > 1:
#        return np.mat([[math.cos(xya[2][0]), -math.sin(xya[2][0]), xya[0][0]], [math.sin(xya[2][0]), math.cos(xya[2][0]), xya[1][0]], [0., 0., 1.]])
#    return np.mat([[math.cos(xya[2]), -math.sin(xya[2]), xya[0]], [math.sin(xya[2]), math.cos(xya[2]), xya[1]], [0., 0., 1.]])


if __name__ == "__main__":
    # ==INIT==
    sparsityN = 2
    active = deque(maxlen=sparsityN)
    toDeactivate = deque(maxlen=sparsityN)
    omega = Omega2()
    xi = Xi()

    # ==CONTROLS DEFINITION==
    # mean = np.array([[0, -1, 2*math.pi]]).transpose()
    mean = np.array([[0., 0., 0.]]).transpose()
    xi.xiVector[0:3] = mean
    # print(xi.xiVector)

    control = (1, 0)
    measurements = np.array([[math.sqrt(2), math.pi / 4, (255, 0, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 0, 0)],
                             [math.sqrt(5), -0.463647, (0, 0, 255)]])

    # measurements = np.array([[1, math.sqrt(2), math.pi / 8, 0],
    #                         [1, math.sqrt(2), math.pi / 8, 0],
    #                         [1, math.sqrt(5), -0.463647, 0],
    #                         [2, math.sqrt(2), -math.pi / 4, 0],
    #                         [3, 1, 0, 0]])


    # ==MAIN BEGIN==
    xi, omega, mean = seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif_measurement_update(xi, omega, mean, measurements)
    mean = seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif_sparsification(xi, omega, mean)

    control = (1, 0)
    measurements = np.array([[math.sqrt(2), -math.pi / 4, (0, 0, 255)]])
    xi, omega, mean = seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif_measurement_update(xi, omega, mean, measurements)
    mean = seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif_sparsification(xi, omega, mean)
    print(mean)

    control = (math.pi/2, math.pi/2)
    measurements = np.array([[1, 0, (0, 255, 0)]])
    xi, omega, mean = seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif_measurement_update(xi, omega, mean, measurements)
    mean = seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif_sparsification(xi, omega, mean)
    print(mean)
