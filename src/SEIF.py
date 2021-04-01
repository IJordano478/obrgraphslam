#!/usr/bin/env python
"""
Online GraphSLAM (SEIF) based on Probabilistic Robotics (2004), Thrun et al


"Sparse Bayesian Information Filters for Localization and Mapping" by Matthew Walter)
issue with overconfidence
"""
from src.frame2d import *
from src.omega import *
from scipy.sparse.linalg import inv
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.sparse import csr_matrix, csc_matrix
from src.simulation.sim_world import *

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import copy
from collections import deque


class SEIF():
    def __init__(self, sparsityN):
        self.omega = Omega2()
        self.xi = Xi()
        self.mean = np.array([[0., 0., 0]]).transpose()

        self.motionNoiseCovariance = [1. ** 2, 1. ** 2, 1. ** 2]
        self.measureNoiseCovariance = [1. ** 2, 1. ** 2]
        self.noiseCovarianceR = np.diag(self.motionNoiseCovariance)
        self.noiseCovarianceQ = np.diag(self.measureNoiseCovariance)

        self.landmarks = 0
        self.sparsityN = sparsityN
        self.active = deque([], maxlen=sparsityN)
        self.toDeactivate = deque([], maxlen=sparsityN)

        fast_motion_update2(np.zeros((3,3)), np.zeros((3,1)), np.zeros((3,1)), np.identity(3), np.zeros((3,3)), np.zeros((3,1)), np.asarray(self.active))

    # def seif_known_correspondence(xi: Xi, omega: Omega2, mean, newControl, newMeasurements):
    #     xi, omega, mean = seif_motion_update(xi, omega, mean, newControl)
    #     xi, omega = seif_measurement_update(xi, omega, mean, newMeasurements)
    #     mean = seif_update_state_estimation(xi, omega, mean)
    #     xi, omega - seif_sparsification(xi, omega)
    #     return xi, omega, mean


    #def seif_motion_update(self, xi, omega, mean, control, timeStep):
    def seif_motion_update(self, control, timeStep):
        v, w = control

        # 2
        Fx = np.concatenate((np.eye(3), np.zeros((3, self.mean.shape[0] - 3))), axis=1)
        FxTranspose = np.transpose(Fx)

        # 3
        w += 0.0000001  # Singularity avoidance
        delta = np.array([[-(v / w) * math.sin(self.mean[2, 0]) + (v / w) * math.sin(self.mean[2, 0] + w * timeStep)],
                          [(v / w) * math.cos(self.mean[2, 0]) - (v / w) * math.cos(self.mean[2, 0] + w * timeStep)],
                          [w * timeStep]])

        #while delta[2, :] > math.pi:
        #    delta[2, :] -= 2 * math.pi
        #while delta[2, :] < -math.pi:
        #    delta[2, :] += 2 * math.pi

        # 4 #ERRATA says this should be negative
        deltaMat = np.array([[0, 0, (v / w) * math.cos(self.mean[2, 0]) - (v / w) * math.cos(self.mean[2, 0] + w * timeStep)],
                             [0, 0, (v / w) * math.sin(self.mean[2, 0]) - (v / w) * math.sin(self.mean[2, 0] + w * timeStep)],
                             [0, 0, 0]])

        print(np.sort(np.asarray(self.active)))

        # self.omega.omegaMatrix, self.xi.xiVector, self.mean = fast_motion_update(self.omega.omegaMatrix, self.xi.xiVector, self.mean, self.noiseCovarianceR,Fx, deltaMat, delta)
        self.omega.omegaMatrix, self.xi.xiVector, self.mean = fast_motion_update2(self.omega.omegaMatrix,
                                                                                self.xi.xiVector, self.mean,
                                                                                self.noiseCovarianceR, deltaMat,
                                                                                delta,
                                                                                np.asarray(self.active))
        return

    def seif_measurement_update(self, measurements, confidence=None):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        if measurements.shape[0] == 0 or measurements.shape[1] == 0:
            return xi, omega, mean

        self.toDeactivate = deque([], maxlen=self.sparsityN)

        # 3
        xiSumMeasurements = np.zeros((1, 1))
        omegaSumMeasurements = np.zeros((1, 1))
        for range_, bearing, signature in measurements:
            t1 = time.time()
            range_ = float(range_)
            bearing = float(bearing)

            z = np.array([[range_], [bearing]])

            # 4,5,6
            # Take measurement and check it against existing landmarks. If its within a certain distance then assume same
            # landmark
            mu_j = mean[0:2, :] + range_ * np.array(
                [[math.cos(bearing + mean[2, 0])], [math.sin(bearing + mean[2, 0])]])

            # This will likely become a correspondence test
            found = False
            i = 0
            for n in range(0, (mean.shape[0] - 3) // 2):
                # TODO include signature match
                if (mu_j[0, 0] - mean[3 + 2 * n, 0]) ** 2 + (mu_j[1, 0] - mean[4 + 2 * n, 0]) ** 2 < 0.5:
                    mu_j = mean[3 + 2 * n:3 + 2 * n + 2, :]
                    found = True
                    # print("matched a landmark")
                    break
                i += 1
            # landmark not seen before, add to map
            if not found:
                mean = np.concatenate((mean, mu_j), axis=0)

            # manage the active landmarks and mark those that need to be deactivated on this timestep
            if i not in self.active:
                if len(self.active) == self.sparsityN and len(self.toDeactivate) != self.sparsityN:
                    self.toDeactivate.append(self.active[0])
                    del(self.active[0])
                if i in self.toDeactivate:
                    del(self.toDeactivate[list(self.toDeactivate).index(i)])
                self.active.append(i)

            # 8
            delta = mu_j - mean[0:2, :]

            # 9
            q = (np.transpose(delta) @ delta)[0][0]

            # 10
            zhat = np.array([[math.sqrt(q)],
                             [math.atan2(delta[1, 0], delta[0, 0]) - mean[2, 0]]])

            # bring angle back into -pi < theta < pi range
            while zhat[1, :] > math.pi:
                zhat[1, :] -= 2 * math.pi
            while zhat[1, :] < -math.pi:
                zhat[1, :] += 2 * math.pi

            omega.omegaMatrix, xi.xiVector = fast_jacobianh_and_updates(omega.omegaMatrix, xi.xiVector, mean, self.noiseCovarianceQ, delta, q, i, z, zhat)

        # 15
        self.xi = xi
        self.omega = omega
        self.mean = mean
        return

    def seif_update_state_estimation(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        # 11
        self.mean = fast_state_estimation(omega.omegaMatrix, xi.xiVector, mean, np.asarray(self.active))
        return

    # def seif_sparsification(self, xi, omega, mean):
    def seif_sparsification(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        self.omega.omegaMatrix, self.xi.xiVector = fast_sparse2(omega.omegaMatrix, xi.xiVector, mean, np.asarray(self.active), np.asarray(self.toDeactivate))
        '''
        toDeactivate = np.asarray(self.toDeactivate)
        active = np.asarray(self.active)
        xi = xi.xiVector
        omega = omega.omegaMatrix
        allIndexes = np.sort((np.concatenate((active, toDeactivate), axis=0)).astype(int))
        np.sort(active)
        np.sort(toDeactivate)
        omegaSize = omega.shape[0]
        omega0Size = allIndexes.shape[0]*2+3

        F_xActDeact = np.zeros((3, omegaSize))
        F_xActDeact[0:3, 0:3] = np.identity(3)

        F_deact = np.zeros((3, omega0Size))

        F_xDeact = np.zeros((3, omega0Size))
        F_xDeact[0:3, 0:3] = np.identity(3)

        F_x = np.zeros((3, omega0Size))
        F_x[0:3, 0:3] = np.identity(3)
        for i in allIndexes:
            F_xActDeact = np.concatenate((F_xActDeact, np.zeros((2, omegaSize))), axis=0)
            F_xActDeact[-2:, 3 + (i * 2):5 + (i * 2)] = np.identity(2)

        if toDeactivate.shape[0] != 0:
            for i in toDeactivate:
                pos = np.where(allIndexes == i)[0][0]

                F_xDeact = np.concatenate((F_xDeact, np.zeros((2, omega0Size))), axis=0)
                F_xDeact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

                F_deact = np.concatenate((F_deact, np.zeros((2, omega0Size))), axis=0)
                F_deact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

        omega_0t = F_xActDeact @ omega @ F_xActDeact.transpose()
        omega1 = -(omega_0t @ F_deact.transpose() @ np.linalg.pinv(F_deact @ omega_0t @ F_deact.transpose()) @ F_deact @ omega_0t)
        omega2 = omega_0t @ F_xDeact.transpose() @ np.linalg.pinv(F_xDeact @ omega_0t @ F_xDeact.transpose()) @ F_xDeact @ omega_0t
        omega3 = -(omega_0t @ F_x.transpose() @ np.linalg.pinv(F_x @ omega_0t @ F_x.transpose()) @ F_x @ omega_0t)
        omega123 = omega1 + omega2 + omega3

        omega123Full = np.zeros((omegaSize, omegaSize))
        omega123Full[0:3, 0:3] = omega123[0:3, 0:3]

        for c in range(0, allIndexes.shape[0]):
            omega123Full[0:3, allIndexes[c] * 2 + 3:allIndexes[c] * 2 + 5] = omega123[0:3, 2 * c + 3:2 * c + 5]
            omega123Full[allIndexes[c] * 2 + 3:allIndexes[c] * 2 + 5, 0:3] = omega123[2 * c + 3:2 * c + 5, 0:3]
            for r in range(0, allIndexes.shape[0]):
                omega123Full[allIndexes[r] * 2 + 3:allIndexes[r] * 2 + 5,
                allIndexes[r] * 2 + 3:allIndexes[r] * 2 + 5] = omega123[2 * r + 3:2 * r + 5, 2 * c + 3:2 * c + 5]

        sparsifiedOmega = omega + omega123Full
        sparsifiedXi = xi + (sparsifiedOmega - omega) @ mean
        self.omega.omegaMatrix = sparsifiedOmega
        self.xi.xiVector = sparsifiedXi
        '''
        return


    #def seif_correspondence_test(self, omega, xi, mean, mj, mk):
    def seif_correspondence_test(self, mj, mk):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        # breakpoint()
        blanketB = np.zeros((0, omega.omegaMatrix.shape[0] - 3))
        overlap = False
        inBlanket = np.empty(0, dtype=int)
        for i in range(3, omega.omegaMatrix.shape[0], 2):
            [found1, found2] = [False, False]
            if not np.allclose(omega.omegaMatrix[3 + mj * 2:5 + mj * 2, i:i + 2], np.zeros((2, 2))):
                found1 = True
            if not np.allclose(omega.omegaMatrix[3 + mk * 2:5 + mk * 2, i:i + 2], np.zeros((2, 2))):
                found2 = True

            if found1 or found2:
                newRow = np.zeros((2, omega.omegaMatrix.shape[0] - 3))
                newRow[:, i - 3:i - 1] = np.identity(2)
                blanketB = np.concatenate((blanketB, newRow), axis=0)
                inBlanket = np.append(inBlanket, (i - 3) // 2)

            if found1 and found2:
                overlap = True

        # 4-7
        if not overlap:
            # 6
            # A* searching the matrix

            # open = [landmarkN, parentN, x, y, F: G+H, G: The cost to get to this node, H: The heuristic to final node]
            open_ = np.array([[float(mj), None, mean[3 + (mj * 2), 0], mean[4 + (mj * 2), 0], 0, 0, math.sqrt(
                (mean[3 + (mj * 2), :] - mean[3 + (mk * 2), :]) ** 2 + (
                            mean[4 + (mj * 2), :] - mean[4 + (mk * 2), :]) ** 2)]])
            closed_ = np.empty((0, 7))

            while True:
                # breakpoint()
                lowestF = (np.where(open_[:, 4] == np.amin(open_[:, 4])))[0]
                closed_ = np.concatenate((closed_, open_[lowestF, :]), axis=0)

                open_ = np.delete(open_, lowestF, 0)

                # Look at all available links to nodes, only do something if a link exists (and it isn't the current node)
                for i in range(3, omega.omegaMatrix.shape[0], 2):
                    if (not np.allclose(omega.omegaMatrix[3 + int(closed_[-1, 0]) * 2:5 + int(closed_[-1, 0]) * 2, i:i + 2],
                                        np.zeros((2, 2)))) and closed_[-1, 0] != (i - 3) // 2:

                        linkLmark = (i - 3) // 2
                        # if node is already in closed_ then ignore it
                        if len(np.where(closed_[:, 0] == linkLmark)[0]) > 0:
                            continue
                        # if new node is not in open, add it with all its characteristics
                        elif len(np.where(open_[:, 0] == linkLmark)[0]) == 0:
                            x = mean[i, 0]
                            y = mean[i + 1, 0]
                            parent = int(closed_[-1, 0])
                            g = closed_[np.where(closed_[:, 0] == parent), 5][0, 0] + math.sqrt(
                                (x - mean[3 + (parent * 2), :]) ** 2 + (y - mean[4 + (parent * 2), :]) ** 2)
                            h = math.sqrt((x - mean[3 + (mk * 2), :]) ** 2 + (y - mean[4 + (mk * 2), :]) ** 2)
                            f = g + h
                            open_ = np.concatenate((open_, np.array([[float(linkLmark), float(parent), x, y, f, g, h]])),
                                                   axis=0)

                        # else new node is already in open, update it if a better route is available
                        else:
                            # index of item in open_
                            openi = int(np.where(open_[:, 0] == linkLmark)[0])
                            x = open_[openi, 2]
                            y = open_[openi, 3]
                            parent = closed_[-1, 0]
                            g = closed_[np.where(closed_[:, 0] == parent), 5][0, 0] + math.sqrt(
                                (x - mean[3 + (int(parent) * 2), :]) ** 2 + (y - mean[4 + (int(parent) * 2), :]) ** 2)

                            if g < open_[openi, 5]:
                                open_[openi, 1] = parent
                                open_[openi, 5] = g
                                open_[openi, 4] = open_[openi, 5] + open_[openi, 6]

                if closed_[-1, 0] == float(mk):
                    break
                if open_.size == 0:
                    break

            # Use the output data from the A* to check whether a path exists and what it is
            if int(closed_[-1, 0]) != mk:
                return 0

            path = np.array([mk])
            while path[-1] != mj:
                next = int(closed_[(np.where(closed_[:, 0] == float(path[-1])))[0], 1])
                path = np.concatenate((path, np.array([next])), axis=0)
            # print("DA shortest path through nodes:\n", path)

            # Create the markov blanket
            blanketB = np.zeros((0, omega.omegaMatrix.shape[0] - 3))
            inBlanket = []
            for lmark in path:
                for i in range(3, omega.omegaMatrix.shape[0], 2):
                    if not np.allclose(omega.omegaMatrix[3 + lmark * 2:5 + lmark * 2, i:2 + i], np.zeros((2, 2))) and (
                            (i - 3) // 2 not in inBlanket):
                        newRow = np.zeros((2, omega.omegaMatrix.shape[0] - 3))
                        newRow[:, i - 3:i - 1] = np.identity(2)
                        blanketB = np.concatenate((blanketB, newRow), axis=0)
                        inBlanket = np.append(inBlanket, (i - 3) // 2)
            np.savetxt("MarkovBlanket.csv", blanketB, fmt='%   1.3f', delimiter=",")

        # print(blanketB)
        # 8

        # 9
        # breakpoint()
        # 10
        localOmega = blanketB @ omega.omegaMatrix[3:, 3:] @ blanketB.transpose()
        localXi = blanketB @ xi.xiVector[3:, :]

        # print(omega.omegaMatrix.round(3))
        # print(localOmega.round(3))
        # print(np.linalg.pinv(localOmega) @ localXi)

        covB = np.linalg.pinv(localOmega)

        # 11
        meanB = covB @ localXi
        # 12

        F_delta = np.zeros((2, localOmega.shape[0]))
        # F_delta[:, 0:2] = np.identity(2)
        inBlanket = np.sort(inBlanket)
        jIndex = np.where(inBlanket == mj)[0][0]
        kIndex = np.where(inBlanket == mk)[0][0]
        F_delta[:, jIndex * 2:jIndex * 2 + 2] = np.identity(2)
        F_delta[:, kIndex * 2:kIndex * 2 + 2] = -np.identity(2)

        # 13
        covDelta = np.linalg.pinv(F_delta @ localOmega @ F_delta.transpose())

        # 14
        meanDelta = covDelta @ F_delta @ localXi
        np.savetxt("covDelta.csv", covDelta, fmt='%   1.3f', delimiter=",")

        # 15
        gaussian = multivariate_normal.pdf(0, meanDelta[:, 0], covDelta, True)
        return gaussian


    #def map_correct(self, omega, xi, mean):
    def map_correct(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        l1 = 3
        l2 = 17
        matchProbability = self.seif_correspondence_test(omega, xi, mean, l1, l2)
        # breakpoint()
        if matchProbability > 0.25:
            F_mjmk = np.zeros((2, omega.omegaMatrix.shape[0] - 3))
            F_mjmk[:, 2 * l1:2 + 2 * l1] = np.identity(2)
            F_mjmk[:, 2 * l2:2 + 2 * l2] = -np.identity(2)
            update = F_mjmk.transpose() @ np.diag([1000000, 1000000]) @ F_mjmk
            np.savetxt("update.csv", update, fmt='%   1.3f', delimiter=",")
            omega.omegaMatrix[3:, 3:] += update

            Fi1 = np.zeros((2, 3 + 2 * l1))
            Fi2 = np.identity(2)
            Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (l1 + 1)))
            Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)
            FiTranspose = np.transpose(Fi)
            mean_it = np.linalg.inv(Fi @ omega.omegaMatrix @ FiTranspose) @ Fi @ (
                    xi.xiVector - (omega.omegaMatrix @ mean) + omega.omegaMatrix @ FiTranspose @ Fi @ mean)
            mean[3 + (l1 * 2):3 + (l1 * 2) + 2, :] = mean_it

            Fi1 = np.zeros((2, 3 + 2 * l2))
            Fi2 = np.identity(2)
            Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (l2 + 1)))
            Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)
            FiTranspose = np.transpose(Fi)
            mean_it = np.linalg.inv(Fi @ omega.omegaMatrix @ FiTranspose) @ Fi @ (
                    xi.xiVector - (omega.omegaMatrix @ mean) + omega.omegaMatrix @ FiTranspose @ Fi @ mean)
            mean[3 + (l2 * 2):3 + (l2 * 2) + 2, :] = mean_it
        return

    def gnss_update(self, z):
        # Get estimate for velocity by gnss
        print(z)
        self.omega.omegaMatrix, self.xi.xiVector, self.mean = fast_gnss_update(self.omega.omegaMatrix,
                                                                                   self.xi.xiVector, self.mean, z,
                                                                                   self.noiseCovarianceQ
                                                                               )
        return

    #def plot_graph_from_mean(self, means):
    def plot_graph_from_mean(self, means):
        means = self.mean

        x = np.array([])
        y = np.array([])

        for i in range(3, means.shape[0], 2):
            x = np.append(x, means[i, 0])
            y = np.append(y, means[i + 1, 0])

        print(x)
        print(y)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(x, y, marker="x")
        plt.show()
        return


    #def plot_graph_from_omega(self, omega, xi, active=False, connections=False, holdprogram=True):
    def plot_graph_from_omega(self, active=False, connections=False, holdprogram=True):
        xi = self.xi
        omega = self.omega

        if type(omega) == Omega2:
            omega = omega.omegaMatrix

        if type(xi) == Xi:
            xi = xi.xiVector

        mean = np.linalg.inv(omega) @ xi

        x = np.array([])
        y = np.array([])

        for i in range(3, mean.shape[0], 2):
            x = np.append(x, mean[i, 0])
            y = np.append(y, mean[i + 1, 0])

        # print(x)
        # print(y)
        fig, ax = plt.subplots(figsize=(10, 10))
        axes = plt.gca()
        axes.set_xlim([-4, 4])
        axes.set_ylim([-2, 4])

        if connections:
            for r in range(3, omega.shape[0], 2):
                for c in range(3, omega.shape[0], 2):
                    if not np.allclose(omega[r:r + 2, c:c + 2], np.zeros((2, 2))):
                        plt.plot([mean[r, 0], mean[c, 0]], [mean[r + 1, 0], mean[c + 1, 0]], color="Black", lw=1)

        if active:
            for i in range(3, omega.shape[0], 2):
                if not np.allclose(omega[0:3, i:i + 2], np.zeros((3, 2))):
                    plt.plot([mean[0, 0], mean[i, 0]], [mean[1, 0], mean[i + 1, 0]], color="Red")

        ax.scatter(x, y, marker="*")
        ax.scatter(mean[0, 0], mean[1, 0], marker="o")

        if holdprogram:
            plt.show()
        else:
            plt.show(block=False)
        return

'''
@jit(nopython=True)
def fast_motion_update(omega, xi, mean, R, Fx, deltaMat, delta):

    FxTranspose = np.transpose(Fx)
    # 5
    psi = FxTranspose @ deltaMat @ Fx
    psiTranspose = np.transpose(psi)

    # 6
    lambda_ = (psiTranspose @ omega) + (omega @ psi) + (psiTranspose @ omega @ psi)
    #lambda_ = np.zeros(omega.omegaMatrix.shape)
    #lambda_[0:3, :] += deltaMat.transpose() @ omega.omegaMatrix[0:3, :]
    #lambda_[:, 0:3] += omega.omegaMatrix[:, 0:3] @ deltaMat
    #lambda_[0:3, 0:3] += deltaMat.transpose() @ omega.omegaMatrix[0:3, 0:3] @ deltaMat

    # 7
    phi = omega + lambda_

    # 8
    kappa = phi @ FxTranspose @ (np.linalg.inv(np.linalg.inv(R) + Fx @ phi @ FxTranspose)) @ Fx @ phi
    #kappa = phi[:, 0:3] @ (np.linalg.inv(np.linalg.inv(self.noiseCovarianceR) + phi[0:3, 0:3])) @ phi[0:3, :]

    # 9
    omega = phi - kappa

    # 10
    xi = xi + ((lambda_ - kappa) @ mean) + (omega @ FxTranspose @ delta)

    # 11
    mean[0:3, :] += delta

    return omega, xi, mean
'''

#@jit(nopython=True)
def fast_motion_update2(omega, xi, mean, R, deltaMat, delta, active):

    Fx_active = np.zeros((3,omega.shape[0]))
    Fx_active[0:3,0:3] = np.identity(3)
    np.sort(active)
    for i in active:
        Fx_active = np.concatenate((Fx_active, np.zeros((2, omega.shape[0]))), axis=0)
        Fx_active[-2:, 2*i+3:2*i+5] = np.identity(2)

    omegaActive = Fx_active @ omega @ Fx_active.transpose()

    Fx = np.concatenate((np.identity(3), np.zeros((3, omegaActive.shape[0]-3))), axis=1)
    FxTranspose = Fx.transpose()
    # 5
    psi = FxTranspose @ deltaMat @ Fx
    psiTranspose = np.transpose(psi)

    # 6
    lambda_ = (psiTranspose @ omegaActive) + (omegaActive @ psi) + (psiTranspose @ omegaActive @ psi)

    # 7
    phi = omegaActive + lambda_

    # 8
    kappa = phi @ FxTranspose @ (np.linalg.inv(np.linalg.inv(R) + Fx @ phi @ FxTranspose)) @ Fx @ phi
    # kappa = phi[:, 0:3] @ (np.linalg.inv(np.linalg.inv(self.noiseCovarianceR) + phi[0:3, 0:3])) @ phi[0:3, :]

    # 9
    omegaActive = phi - kappa

    # 10
    xiActive = (Fx_active @ xi) + ((lambda_ - kappa) @ (Fx_active @ mean)) + (omegaActive @ FxTranspose @ delta)

    omega[0:3, 0:3] = omegaActive[0:3, 0:3]
    xi[0:3, :] = xiActive[0:3, :]

    for c in range(0, active.shape[0]):
        omega[0:3, active[c] * 2 + 3:active[c] * 2 + 5] = omegaActive[0:3, 2 * c + 3:2 * c + 5]
        omega[active[c] * 2 + 3:active[c] * 2 + 5, 0:3] = omegaActive[2 * c + 3:2 * c + 5, 0:3]
        xi[active[c] * 2 + 3:active[c] * 2 + 5, :] = xiActive[c * 2 + 3:c * 2 + 5, :]
        for r in range(0, active.shape[0]):
            omega[active[r] * 2 + 3:active[r] * 2 + 5, active[c] * 2 + 3:active[c] * 2 + 5] = omegaActive[2 * r + 3:2 * r + 5, 2 * c + 3:2 * c + 5]

    # 11
    mean[0:3, :] += delta
    return omega, xi, mean

'''
@jit(nopython=True)
def fast_sparse(omega, xi, mean, active, toDeactivate):
    omegaSize = omega.shape

    F_xActDeact = np.zeros(omegaSize)
    F_xActDeact[0:3, 0:3] = np.identity(3)

    F_deact = np.zeros(omegaSize)

    F_xDeact = np.zeros(omegaSize)
    F_xDeact[0:3, 0:3] = np.identity(3)

    F_x = np.zeros(omegaSize)
    F_x[0:3, 0:3] = np.identity(3)

    for i in active:
        F_xActDeact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    for i in toDeactivate:
        F_xActDeact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)
        F_xDeact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)
        F_deact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    omega_0t = F_xActDeact @ omega @ F_xActDeact
    omega1 = -(omega_0t @ F_deact @ np.linalg.pinv(F_deact @ omega_0t @ F_deact) @ F_deact @ omega_0t)
    omega2 = omega_0t @ F_xDeact @ np.linalg.pinv(F_xDeact @ omega_0t @ F_xDeact) @ F_xDeact @ omega_0t
    omega3 = -(omega @ F_x @ np.linalg.pinv(F_x @ omega @ F_x) @ F_x @ omega)
    sparsifiedOmega = omega + omega1 + omega2 + omega3
    sparsifiedXi = xi + (sparsifiedOmega - omega) @ mean
    return sparsifiedOmega, sparsifiedXi
'''

#@jit(nopython=True)
def fast_sparse2(omega, xi, mean, active, toDeactivate):
    allIndexes = np.sort((np.concatenate((active, toDeactivate), axis=0)).astype(np.int32))
    np.sort(active)
    np.sort(toDeactivate)
    omegaSize = omega.shape[0]
    omega0Size = allIndexes.shape[0] * 2 + 3

    F_xActDeact = np.zeros((3, omegaSize))
    F_xActDeact[0:3, 0:3] = np.identity(3)

    F_deact = np.zeros((3, omega0Size))

    F_xDeact = np.zeros((3, omega0Size))
    F_xDeact[0:3, 0:3] = np.identity(3)

    F_x = np.zeros((3, omega0Size))
    F_x[0:3, 0:3] = np.identity(3)
    for i in allIndexes:
        F_xActDeact = np.concatenate((F_xActDeact, np.zeros((2, omegaSize))), axis=0)
        F_xActDeact[-2:, 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    if toDeactivate.shape[0] != 0:
        for i in toDeactivate:
            pos = np.where(allIndexes == i)[0][0]

            F_xDeact = np.concatenate((F_xDeact, np.zeros((2, omega0Size))), axis=0)
            F_xDeact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

            F_deact = np.concatenate((F_deact, np.zeros((2, omega0Size))), axis=0)
            F_deact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

    omega_0t = F_xActDeact @ omega @ F_xActDeact.transpose()
    omega1 = -(omega_0t @ F_deact.transpose() @ np.linalg.pinv(F_deact @ omega_0t @ F_deact.transpose()) @ F_deact @ omega_0t)
    omega2 = omega_0t @ F_xDeact.transpose() @ np.linalg.pinv(F_xDeact @ omega_0t @ F_xDeact.transpose()) @ F_xDeact @ omega_0t
    omega3 = -(omega_0t @ F_x.transpose() @ np.linalg.pinv(F_x @ omega_0t @ F_x.transpose()) @ F_x @ omega_0t)
    omega123 = omega1 + omega2 + omega3

    omega123Full = np.zeros((omegaSize, omegaSize))
    omega123Full[0:3, 0:3] = omega123[0:3, 0:3]

    for c in range(0, allIndexes.shape[0]):
        omega123Full[0:3, allIndexes[c] * 2 + 3:allIndexes[c] * 2 + 5] = omega123[0:3, 2 * c + 3:2 * c + 5]
        omega123Full[allIndexes[c] * 2 + 3:allIndexes[c] * 2 + 5, 0:3] = omega123[2 * c + 3:2 * c + 5, 0:3]
        for r in range(0, allIndexes.shape[0]):
            omega123Full[allIndexes[r] * 2 + 3:allIndexes[r] * 2 + 5, allIndexes[r] * 2 + 3:allIndexes[r] * 2 + 5] = omega123[2 * r + 3:2 * r + 5, 2 * c + 3:2 * c + 5]

    sparsifiedOmega = omega + omega123Full
    sparsifiedXi = xi + (sparsifiedOmega - omega) @ mean
    return sparsifiedOmega, sparsifiedXi

#@jit(nopython=True)
def fast_jacobianh_and_updates(omega, xi, mean, Q, delta, q, i, z, zhat):
    h1 = np.array([[-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0], 0],
                   [delta[1, 0], -delta[0, 0], -q]])

    h2 = np.zeros((2, 2 * i))

    h3 = np.array([[math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0]],
                   [-delta[1, 0], delta[0, 0]]])

    h4 = np.zeros((2, (mean.shape[0] - 3) - 2 * (i + 1)))

    h_it = 1 / q * np.concatenate((h1, h2, h3, h4), axis=1)

    xiUpdate = np.transpose(h_it) @ np.linalg.inv(Q) @ (z - zhat + (h_it @ mean))

    # resize array with new columns and rows as zeros
    if xiUpdate.shape > xi.shape:
        newArray = np.zeros(xiUpdate.shape)
        newArray[0:xi.shape[0], 0:xi.shape[1]] = xi
        xi = newArray

    omegaUpdate = np.transpose(h_it) @ np.linalg.inv(Q) @ h_it
    if omegaUpdate.shape > omega.shape:
        newMatrix = np.zeros(omegaUpdate.shape)
        newMatrix[0:omega.shape[0], 0:omega.shape[0]] = omega
        omega = newMatrix

    xi += xiUpdate
    omega += omegaUpdate
    return omega, xi

#@jit(nopython=True)
def fast_state_estimation(omega, xi, mean, active):
    # 2
    # for small number of active map features
    for i in active:
        Fi1 = np.zeros((2, 3 + 2 * i))
        Fi2 = np.identity(2)
        Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (i + 1)))
        Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)

        mean_it = np.linalg.inv(Fi @ omega @ Fi.transpose()) @ Fi @ (xi - (omega @ mean) + omega @ Fi.transpose() @ Fi @ mean)
        mean[3 + (i * 2):3 + (i * 2) + 2, :] = mean_it

    # 9
    Fx = np.concatenate((np.identity(3), np.zeros((3, mean.shape[0] - 3))), axis=1)

    # 10
    updatedMean = np.linalg.inv(Fx @ omega @ Fx.transpose()) @ Fx @ (xi - (omega @ mean) + (omega @ Fx.transpose() @ Fx @ mean))
    mean[0:3, :] = updatedMean
    return mean


def fast_gnss_update(omega, xi, mean, z, Q):
    h = mean[0:2, :]
    grad_h_x = np.array([[1., 0., 0.],
                         [0., 1., 0.]])
    grad_h_w = np.array([[1., 0.],
                         [0., 1.]])

    Fx = np.concatenate((np.identity(3), np.zeros((3, omega.shape[0] - 3))), axis=1)
    cov_x = np.linalg.pinv(Fx @ omega @ Fx.transpose())

    s1 = grad_h_w @ Q @ grad_h_w.transpose()
    s2 = (grad_h_x @ cov_x[0:3, 0:3] @ grad_h_x.transpose())
    s = s1 + s2

    w = cov_x[0:3, 0:3] @ grad_h_x.transpose() @ np.linalg.pinv(s)

    mean[0:3, :] = mean[0:3, :] + (w @ (z - h))
    cov_x[0:3, 0:3] += - w @ s @ w.transpose()
    omega_x = np.linalg.pinv(cov_x)
    # self.omega.omegaMatrix[0:3,:] = omega_x[0:3,:]
    # self.omega.omegaMatrix[:,0:3] = omega_x[:,0:3]
    omega[0:3, 0:3] = omega_x[0:3, 0:3]

    xi = omega @ mean
    return omega, xi, mean

# if __name__ == "__main__":
def run_seif_testing():
    # ==INIT==
    sparsityN = 5
    active = deque([], maxlen=sparsityN)
    toDeactivate = deque([], maxlen=sparsityN)
    omega = Omega2()
    xi = Xi()
    time = 0
    seif = SEIF()

    # ==CONTROLS DEFINITION==
    # mean = np.array([[0, -1, 2*math.pi]]).transpose()
    mean = np.array([[0., 0., 0.]]).transpose()
    xi.xiVector[0:3] = mean
    # print(xi.xiVector)

    # ==MAIN BEGIN==
    # do initial observation for t = 0
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    print("Time:", time)
    print("Mean:\n", mean.round(4))

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    # plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    seif.plot_graph_from_omega(omega, xi, active=True, connections=True, holdprogram=False)
    for m1 in range(0, (omega.omegaMatrix.shape[0] - 3) // 2 - 1):
        for m2 in range(0, (omega.omegaMatrix.shape[0] - 3) // 2 - 1):
            if m1 != m2:
                print("Comparing", m1, "to", m2, ":", seif.seif_correspondence_test(omega, xi, mean, m1, m2))
    np.savetxt("omega.csv", omega.omegaMatrix, fmt='%   1.3f', delimiter=",")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True, holdprogram=True)
    breakpoint()

    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 165, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (255, 165, 0)]])
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    print("Time:", time)
    print("Mean:\n", mean.round(4))
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    print(omega.omegaMatrix.round(1))
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (2, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (2, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)

    control = (2, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 165, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1.5), -math.pi / 2, (255, 165, 0)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    control = (1, 0)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    print(omega.omegaMatrix.round(1))
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    control = (math.pi / 2, math.pi / 2)
    measurements = np.array([[math.sqrt(1), math.pi / 2, (255, 255, 0)],
                             [math.sqrt(2), math.pi / 4, (255, 255, 0)],
                             [math.sqrt(2), -math.pi / 4, (0, 0, 255)],
                             [math.sqrt(1), -math.pi / 2, (0, 0, 255)]])
    xi, omega, mean = seif.seif_motion_update(xi, omega, mean, control)
    xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
    mean = seif.seif_update_state_estimation(xi, omega, mean)
    xi, omega = seif.seif_sparsification(xi, omega, mean)
    time += 1
    print("Time:", time)
    print("Mean:\n", mean.round(3), "\n")

    print((np.linalg.inv(omega.omegaMatrix) @ xi.xiVector).round(2))
    seif.plot_graph_from_omega(omega, xi, active=True, connections=True)
    # plot_graph_from_omega(omega.omegaMatrix, mean, active=False, connections=True)

    # np.set_printoptions(precision=2, suppress=True)
    print(omega.omegaMatrix)
    np.savetxt("omega.csv", omega.omegaMatrix, fmt='%   1.3f', delimiter=",")
    # breakpoint()

    np.savetxt("omega.csv", omega.omegaMatrix, fmt='%   1.3f', delimiter=",")

    seif.plot_graph_from_omega(omega.omegaMatrix, xi, active=True, connections=True)
    print((np.linalg.inv(omega.omegaMatrix) @ xi.xiVector).round(2))
    print(mean.round(3))
    # seif_correspondence_test(omega, xi, mean, 3, 16)


def run_seif(w: SimWorld, finished):
    global active
    global toDeactivate
    global landmarks

    # ==INIT==
    sparsityN = 10
    active = deque([], maxlen=sparsityN)
    toDeactivate = deque([], maxlen=sparsityN)
    omega = Omega2()
    xi = Xi()
    timestep = 1
    seif = SEIF()

    # ==CONTROLS DEFINITION==
    mean = np.array([[8., 5., math.pi / 2]]).transpose()
    xi.xiVector[0:3] = mean

    time.sleep(2)

    while not finished.is_set():
        t0 = time.time()

        imu = w.sensor_imu()
        gps = w.sensor_gps()
        measurements = w.sensor_camera()
        print("imu:", imu)
        print("gps:", gps)
        print("camera:", measurements)
        xi, omega, mean = seif.seif_motion_update(xi, omega, mean, imu)
        xi, omega, mean = seif.seif_measurement_update(xi, omega, mean, measurements)
        mean = seif.seif_update_state_estimation(xi, omega, mean)
        xi, omega = seif.seif_sparsification(xi, omega, mean)
        np.savetxt("omega.csv", omega.omegaMatrix, fmt='%   1.3f', delimiter=",")
        np.savetxt("xi.csv", xi.xiVector, fmt='%   1.3f', delimiter=",")
        np.savetxt("mean.csv", mean[3:], fmt='%   1.3f', delimiter=",")

        landmarks = mean[3:]

        # simulate the rest of the timestep if code is too fast
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < timestep:
            time.sleep(seif.timeStep - timeTaken)
