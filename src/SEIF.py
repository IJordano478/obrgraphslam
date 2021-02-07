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

timeStep = 1
motionNoiseCovariance = [1.**2, 1.**2, 1.**2]
measureNoiseCovariance = [1.**2, 1.**2]

noiseCovarianceR = np.diag(motionNoiseCovariance)
noiseCovarianceQ = np.diag(measureNoiseCovariance)

landmarks = np.empty((0, 2))


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
    nLandmarks = 0  # Irrelevant, set to large value to avoid resize

    # 2
    Fx = np.concatenate((np.eye(3), np.zeros((3, 2 * nLandmarks))), axis=1)
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
    mean = mean + delta

    print("mean should be:\n", mean.round(3))
    print("Mean from information matrix and vector:\n", (np.linalg.inv(omega.omegaMatrix[0:3, 0:3])@xi.xiVector).round(3), "\n\n")
    # 12
    # breakpoint()
    return xi, omega, mean


def seif_measurement_update(xi, omega, mean, measurements):
    #breakpoint()

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
        global landmarks
        found = False
        i = 0
        for x, y in landmarks:
            # TODO include signature match
            if (mu_j[0, 0] - x)**2 + (mu_j[1, 0] - y)**2 < 1:
                mu_j = np.array([[x], [y]])
                found = True
                break
            i += 1

        # landmark not seen before, add to map
        if not found:
            if landmarks.shape[0] == 0:
                landmarks = np.transpose(mu_j)
            else:
                landmarks = np.concatenate((landmarks, np.transpose(mu_j)), axis=0)

        # 8
        delta = mu_j - mean[0:2, :]

        # 9
        q = (np.transpose(delta) @ delta)[0][0]

        # 10
        a2 = math.atan2(delta[1, 0], delta[0, 0])
        zhat = np.array([[math.sqrt(q)],
                         [math.atan2(delta[1, 0], delta[0,0]) - mean[2, 0]]])

        print(mean.round(3))
        # 11
        # jacobian H must be made of 4 different matrix components; the pose, 0s, the landmark, 0s.
        h1 = np.array([[-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0], 0],
                       [delta[1, 0], -delta[0, 0], -q]])

        h2 = np.zeros((2, 2 * (i+1) - 2))

        h3 = np.array([[math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0]],
                       [-delta[1, 0], delta[0, 0]]])

        h4 = np.zeros((2, 2*landmarks.shape[0] - 2*(i+1)))

        h_it = 1 / q * np.concatenate((h1, h2, h3, h4), axis=1)

        meanXiUpdate = np.zeros((h_it.shape[1], 1))
        meanXiUpdate[0:3, :] = mean
        meanXiUpdate[3+(2*i):3+(2*i)+2, :] = mu_j

        # xiUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat +
        # breakpoint()
        a = (h_it @ meanXiUpdate)
        b = (z - zhat).round(3)
        c = (z - zhat + (h_it @ meanXiUpdate))
        xiUpdate = np.transpose(h_it) @ np.linalg.inv(noiseCovarianceQ) @ (z - zhat + (h_it @ meanXiUpdate))
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

        print("xiSum\n:",xiSumMeasurements)
        print("omegaSum\n:", omegaSumMeasurements)
        print("\n")

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
    # breakpoint()
    print((np.linalg.inv(omega.omegaMatrix) @ xi.xiVector).round(3))
    return xi, omega


def seif_update_state_estimation(xi, omega, mean):

    return


def seif_sparsification(xi, omega, mean):
    # 2
    nEntries = omega.nPoses+omega.nLmarks
    f_m0 = np.zeros((3, nEntries))
    f_m0[:, 3:6] = np.identity(3)

    f_xm0 = np.zeros((6, nEntries))
    f_xm0[0:3, 0:3] = np.identity(3)
    f_xm0[3:6, 3:6] = np.identity(3)

    f_x = np.zeros((3, nEntries))
    f_x[:0:3] = np.identity(3)

    #infoMatrix0 =

    #infoMatrix1 =

    return


def seif_correspondence_test():
    return


def xya_to_matrix(xya):
    if len(xya.shape) > 1:
        return np.mat([[math.cos(xya[2][0]), -math.sin(xya[2][0]), xya[0][0]], [math.sin(xya[2][0]), math.cos(xya[2][0]), xya[1][0]], [0., 0., 1.]])
    return np.mat([[math.cos(xya[2]), -math.sin(xya[2]), xya[0]], [math.sin(xya[2]), math.cos(xya[2]), xya[1]], [0., 0., 1.]])


if __name__ == "__main__":

    omega = Omega2()
    xi = Xi()

    #mean = np.array([[0, -1, 2*math.pi]]).transpose()
    mean = np.array([[0, -1, 0]]).transpose()
    xi.xiVector[0:3] = mean
    print(xi.xiVector)

    control = (math.pi/2, math.pi/2)

    #landmarks = np.array([[10, math.pi / 4, (255, 0, 0)],
    #                      [5, -math.pi / 4, (0, 255, 0)]])

    measurementsTest = np.array([[10, math.pi / 4, (255, 0, 0)],
                                 [5, -math.pi / 4, (0, 255, 0)],
                                 [5.0000001, -math.pi / 4, (0, 255, 0)]])


    xi, omega, mean = seif_motion_update(xi, omega, mean, control)

    #breakpoint()
    seif_measurement_update(xi, omega, mean, measurementsTest)
    breakpoint()
    mean = seif_update_state_estimation(xi, omega, mean)
    breakpoint()
