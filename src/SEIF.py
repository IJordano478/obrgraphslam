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
    def __init__(self, sparsity_n):
        self.rss = 3  # Robot state size: {x,y,theta}
        self.lss = 3  # Landmark state size: {x,y,colour}
        self.omega = Omega(rss=self.rss, lss=self.lss)
        self.xi = Xi()
        self.mean = np.array([[0., 0., 0]]).transpose()

        self.motion_noise_covariance = [.001 ** 2, .001 ** 2, .001 ** 2]
        self.measure_noise_covariance = [.1 ** 2, .1 ** 2, .00001 ** 2]
        self.gnss_noise_covariance = [.001 ** 2, .001 ** 2]
        self.noise_covariance_r = np.diag(self.motion_noise_covariance)
        self.noise_covariance_q = np.diag(self.measure_noise_covariance)
        self.gnss_noise_covariance_q = np.diag(self.gnss_noise_covariance)

        self.landmarks = 0
        self.sparsity_n = sparsity_n
        self.active = deque([], maxlen=sparsity_n)
        self.to_deactivate = deque([], maxlen=sparsity_n)
        self.signature_map = {"Blue": 1, "blue": 1, "Yellow": 2, "yellow": 2, "Orange": 3, "orange": 3}

        #fast_motion_update2(np.zeros((3,3)), np.zeros((3,1)), np.zeros((3,1)), np.identity(3), np.zeros((3,3)), np.zeros((3,1)), np.asarray(self.active))

    # def seif_known_correspondence(xi: Xi, omega: Omega2, mean, newControl, newMeasurements):
    #     xi, omega, mean = seif_motion_update(xi, omega, mean, newControl)
    #     xi, omega = seif_measurement_update(xi, omega, mean, newMeasurements)
    #     mean = seif_update_state_estimation(xi, omega, mean)
    #     xi, omega - seif_sparsification(xi, omega)
    #     return xi, omega, mean

    def seif_motion_update(self, control, time_step):

        omega = self.omega.omega_matrix
        xi = self.xi.xi_vector
        mean = self.mean
        active = np.asarray(self.active)

        v, w = control

        # 2
        # Fx = np.concatenate((np.eye(3), np.zeros((3, mean.shape[0] - 3))), axis=1)
        # Fx_transpose = np.transpose(Fx)

        # 3
        # w += 0.00000000001  # Singularity avoidance
        w += 1*10**(-10)
        delta = np.array([[-(v / w) * math.sin(mean[2, 0]) + (v / w) * math.sin(mean[2, 0] + w * time_step)],
                          [(v / w) * math.cos(mean[2, 0]) - (v / w) * math.cos(mean[2, 0] + w * time_step)],
                          [w * time_step]])

        # delta[2,:] = (delta[2,:] + np.pi) % (2 * np.pi) - np.pi

        # 4 #ERRATA says this should be negative
        delta_mat = np.array([[0, 0, (v / w) * math.cos(mean[2, 0]) - (v / w) * math.cos(mean[2, 0] + w * time_step)],
                              [0, 0, (v / w) * math.sin(mean[2, 0]) - (v / w) * math.sin(mean[2, 0] + w * time_step)],
                              [0, 0, 0]])

        Fx_active = np.zeros((self.rss, omega.shape[0]))
        Fx_active[0:self.rss, 0:self.rss] = np.identity(self.rss)
        Fx_reducer = np.identity(omega.shape[0])
        Fx_reducer[0:self.rss, 0:self.rss] = np.zeros(self.rss)
        active = np.sort(active)
        for i in active:
            Fx_active = np.concatenate((Fx_active, np.zeros((self.lss, omega.shape[0]))), axis=0)
            Fx_active[-self.lss:, self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss] = np.identity(self.lss)
            Fx_reducer[self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss, self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss] = np.zeros((3, 3))
            #modified_identity = np.identity(self.lss)
            #modified_identity[-1, -1] = 0.000001
            #Fx_active[-self.lss:, self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss] = modified_identity
        omega_active = Fx_active @ omega @ Fx_active.transpose()

        Fx = np.concatenate((np.identity(self.rss), np.zeros((self.rss, omega_active.shape[0] - self.rss))), axis=1)
        Fx_transpose = Fx.transpose()
        # 5
        psi = Fx_transpose @ delta_mat @ Fx
        psi_transpose = np.transpose(psi)

        # 6
        lambda_ = (psi_transpose @ omega_active) + (omega_active @ psi) + (psi_transpose @ omega_active @ psi)

        # 7
        phi = omega_active + lambda_

        # 8
        kappa = phi @ Fx_transpose @ (
            np.linalg.inv(np.linalg.inv(self.noise_covariance_r) + Fx @ phi @ Fx_transpose)) @ Fx @ phi
        # kappa = phi[:, 0:3] @ (np.linalg.inv(np.linalg.inv(self.noise_covariance_r) + phi[0:3, 0:3])) @ phi[0:3, :]

        # 9
        omega_active = phi - kappa

        # 10
        xi_active = (Fx_active @ xi) + ((lambda_ - kappa) @ (Fx_active @ mean)) + (omega_active @ Fx_transpose @ delta)

        # Method 1: Alternate method for reconstruction below. Uses in-built matrix multiplication
        # Access efficiency: (RSS + LSS * N)^2
        # Time analysis showed this method was slower on average than
        #omega = omega - Fx_active.transpose() @ Fx_active @ omega @ Fx_active.transpose() @ Fx_active + Fx_active.transpose() @ omega_active @ Fx_active
        #xi = xi - Fx_active.transpose() @ Fx_active @ xi + Fx_active.transpose() @ xi_active

        # Method 2: Index based method for reconstruction. Always accesses at most (RSS + LSS * N_active)^2 elements
        omega[0:self.rss, 0:self.rss] = omega_active[0:self.rss, 0:self.rss]
        xi[0:self.rss, :] = xi_active[0:self.rss, :]
        for c in range(0, active.shape[0]):
            lower_r_1 = 0
            upper_r_1 = self.rss
            lower_c_1 = self.rss + (self.lss * active[c])
            upper_c_1 = lower_c_1 + self.lss
            lower_r_2 = 0
            upper_r_2 = self.rss
            lower_c_2 = self.rss + (self.lss * c)
            upper_c_2 = lower_c_2 + self.lss
            omega[:upper_r_1, lower_c_1:upper_c_1] = omega_active[:upper_r_2, lower_c_2:upper_c_2]

            lower_r_1 = self.rss + (self.lss * active[c])
            upper_r_1 = lower_r_1 + self.lss
            lower_c_1 = 0
            upper_c_1 = self.rss
            lower_r_2 = self.rss + (self.lss * c)
            upper_r_2 = lower_r_2 + self.lss
            lower_c_2 = 0
            upper_c_2 = self.rss
            omega[lower_r_1:upper_r_1, :upper_c_1] = omega_active[lower_r_2:upper_r_2, :upper_c_2]

            # Reuse row locations for xi, no column indexing needed as xi is a vector
            xi[lower_r_1:upper_r_1, :] = xi_active[lower_r_2:upper_r_2, :]

            for r in range(0, active.shape[0]):
                lower_r_1 = self.rss + (self.lss * active[r])
                upper_r_1 = lower_r_1 + self.lss
                lower_c_1 = self.rss + (self.lss * active[c])
                upper_c_1 = lower_c_1 + self.lss
                lower_r_2 = self.rss + (self.lss * r)
                upper_r_2 = lower_r_2 + self.lss
                lower_c_2 = self.rss + (self.lss * c)
                upper_c_2 = lower_c_2 + self.lss
                omega[lower_r_1:upper_r_1, lower_c_1:upper_c_1] = omega_active[lower_r_2:upper_r_2, lower_c_2:upper_c_2]

        # 11
        mean[0:self.rss, :] += delta

        self.omega.omega_matrix = omega
        self.xi.xi_vector = xi
        self.mean = mean

        return


    def seif_measurement_update(self, measurements, confidence=None):
        xi = self.xi.xi_vector
        omega = self.omega.omega_matrix
        mean = self.mean

        if measurements.shape[0] == 0 or measurements.shape[1] == 0:
            return xi, omega, mean

        self.to_deactivate = deque([], maxlen=self.sparsity_n)

        # 3
        for range_, bearing, signature in measurements:
            range_ = float(range_)
            bearing = float(bearing)
            signature = self.signature_map[signature]

            z = np.array([[range_], [bearing], [signature]])

            # 4,5,6
            # Take measurement and check it against existing landmarks. If its within a certain distance then assume
            # same
            # landmark
            mu_j = mean[0:2, :] + range_ * np.array(
                [[math.cos(bearing + mean[2, 0])], [math.sin(bearing + mean[2, 0])]])
            mu_j = np.concatenate((mu_j, np.array([[signature]])), axis=0)

            # This will likely become a correspondence test
            found = False
            i = 0
            for n in range(0, (mean.shape[0] - self.rss) // self.lss):
                # TODO include signature match
                x_index = self.rss + self.lss * n
                y_index = x_index + 1
                s_index = x_index + 2
                if ((mu_j[0, 0] - mean[x_index, 0]) ** 2 + (mu_j[1, 0] - mean[y_index, 0]) ** 2 < 1) and (mean[s_index, 0] == mu_j[2, 0]):
                    mu_j = mean[x_index:x_index+self.lss, :]
                    found = True
                    # print("matched a landmark")
                    break
                i += 1
            # landmark not seen before, add to map
            if not found:
                mean = np.concatenate((mean, mu_j), axis=0)

            # manage the active landmarks and mark those that need to be deactivated on this time_step
            if i not in self.active:
                if len(self.active) == self.sparsity_n and len(self.to_deactivate) != self.sparsity_n:
                    self.to_deactivate.append(self.active[0])
                    del (self.active[0])
                if i in self.to_deactivate:
                    del (self.to_deactivate[list(self.to_deactivate).index(i)])
                self.active.append(i)

            # 8
            delta = mu_j[0:2, :] - mean[0:2, :]

            # 9
            q = (np.transpose(delta) @ delta)[0][0]

            # 10
            zhat = np.array([[math.sqrt(q)],
                             [math.atan2(delta[1, 0], delta[0, 0]) - mean[2, 0]],
                             [signature]])

            zhat[1, :] = (zhat[1, :] + np.pi) % (2 * np.pi) - np.pi

            h1 = np.array([[-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0], 0],
                           [delta[1, 0], -delta[0, 0], -q],
                           [0, 0, 0]])

            h2 = np.zeros((3, self.lss * i))

            h3 = np.array([[math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0], 0],
                           [-delta[1, 0], delta[0, 0], 0],
                           [0, 0, q]])

            h4 = np.zeros((3, (mean.shape[0] - self.rss) - self.lss * (i + 1)))

            h_it = 1 / q * np.concatenate((h1, h2, h3, h4), axis=1)

            xi_update = np.transpose(h_it) @ np.linalg.inv(self.noise_covariance_q) @ (z - zhat + (h_it @ mean))

            # resize array with new columns and rows as zeros
            if xi_update.shape > xi.shape:
                new_array = np.zeros(xi_update.shape)
                new_array[0:xi.shape[0], 0:xi.shape[1]] = xi
                xi = new_array

            omega_update = np.transpose(h_it) @ np.linalg.inv(self.noise_covariance_q) @ h_it
            if omega_update.shape > omega.shape:
                new_matrix = np.zeros(omega_update.shape)
                new_matrix[0:omega.shape[0], 0:omega.shape[0]] = omega
                omega = new_matrix
            #breakpoint()

            xi += xi_update
            omega += omega_update

        # 15
        self.xi.xi_vector = xi
        self.omega.omega_matrix = omega
        self.mean = mean
        return


    def seif_update_state_estimation(self):
        xi = self.xi.xi_vector
        omega = self.omega.omega_matrix
        mean = self.mean
        active = np.asarray(self.active)

        if len(self.active) == 0:
            return

        # for small number of active map features
        for i in active:
            Fi1 = np.zeros((self.lss, self.rss + self.lss * i))
            Fi2 = np.identity(self.lss)
            Fi3 = np.zeros((self.lss, (mean.shape[0] - self.rss) - self.lss * (i + 1)))
            Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)

            mean_it = np.linalg.pinv(Fi @ omega @ Fi.transpose()) @ Fi @ (
                    xi - (omega @ mean) + omega @ Fi.transpose() @ Fi @ mean)
            mean[self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss, :] = mean_it

        Fx = np.concatenate((np.identity(self.rss), np.zeros((self.rss, mean.shape[0] - self.rss))), axis=1)

        updated_mean = np.linalg.inv(Fx @ omega @ Fx.transpose()) @ Fx @ (
                xi - (omega @ mean) + (omega @ Fx.transpose() @ Fx @ mean))
        mean[0:self.rss, :] = updated_mean

        # 11
        self.mean = mean
        return


    # def seif_sparsification(self, xi, omega, mean):
    def seif_sparsification(self):
        xi = self.xi.xi_vector
        omega = self.omega.omega_matrix
        mean = self.mean
        active = np.asarray(self.active)
        to_deactivate = np.asarray(self.to_deactivate)

        all_indexes = np.sort((np.concatenate((active, to_deactivate), axis=0)).astype(np.int32))
        #active = np.sort(active)
        #to_deactivate = np.sort(to_deactivate)
        omega_size = omega.shape[0]
        omega0Size = self.rss + self.lss * all_indexes.shape[0]

        F_x_act_deact = np.zeros((self.rss, omega_size))
        F_x_act_deact[0:self.rss, 0:self.rss] = np.identity(self.rss)

        F_deact = np.zeros((self.rss, omega0Size))

        F_x_deact = np.zeros((self.rss, omega0Size))
        F_x_deact[0:self.rss, 0:self.rss] = np.identity(self.rss)

        F_x = np.zeros((self.rss, omega0Size))
        F_x[0:self.rss, 0:self.rss] = np.identity(self.rss)
        for i in all_indexes:
            F_x_act_deact = np.concatenate((F_x_act_deact, np.zeros((self.lss, omega_size))), axis=0)
            F_x_act_deact[-self.lss:, self.rss + (self.lss * i):self.rss + (self.lss * i) + self.lss] = np.identity(self.lss)

        if to_deactivate.shape[0] != 0:
            for i in to_deactivate:
                pos = np.where(all_indexes == i)[0][0]

                F_x_deact = np.concatenate((F_x_deact, np.zeros((self.lss, omega0Size))), axis=0)
                F_x_deact[-self.lss:, self.rss + (self.lss * pos):self.rss + (self.lss * pos) + self.lss] = np.identity(self.lss)

                F_deact = np.concatenate((F_deact, np.zeros((self.lss, omega0Size))), axis=0)
                F_deact[-self.lss:, self.rss + (self.lss * pos):self.rss + (self.lss * pos) + self.lss] = np.identity(self.lss)

        omega_0t = F_x_act_deact @ omega @ F_x_act_deact.transpose()
        omega1 = -(omega_0t @ F_deact.transpose() @ np.linalg.pinv(
            F_deact @ omega_0t @ F_deact.transpose()) @ F_deact @ omega_0t)
        omega2 = omega_0t @ F_x_deact.transpose() @ np.linalg.pinv(
            F_x_deact @ omega_0t @ F_x_deact.transpose()) @ F_x_deact @ omega_0t
        omega3 = -(omega_0t @ F_x.transpose() @ np.linalg.pinv(F_x @ omega_0t @ F_x.transpose()) @ F_x @ omega_0t)
        omega123 = omega1 + omega2 + omega3

        omega123_full = np.zeros((omega_size, omega_size))
        omega123_full[0:self.rss, 0:self.rss] = omega123[0:self.rss, 0:self.rss]

        for c in range(0, all_indexes.shape[0]):
            lower_r_1 = 0
            upper_r_1 = self.rss
            lower_c_1 = self.rss + self.lss * all_indexes[c]
            upper_c_1 = lower_c_1 + self.lss
            lower_r_2 = 0
            upper_r_2 = self.rss
            lower_c_2 = self.rss + self.lss * c
            upper_c_2 = lower_c_2 + self.lss
            omega123_full[lower_r_1:upper_r_1, lower_c_1:upper_c_1] = omega123[lower_r_2:upper_r_2, lower_c_2:upper_c_2]

            lower_r_1 = self.rss + self.lss * all_indexes[c]
            upper_r_1 = lower_r_1 + self.lss
            lower_c_1 = 0
            upper_c_1 = self.rss
            lower_r_2 = self.rss + self.lss * c
            upper_r_2 = lower_r_2 + self.lss
            lower_c_2 = 0
            upper_c_2 = self.rss
            omega123_full[lower_r_1:upper_r_1, lower_c_1:upper_c_1] = omega123[lower_r_2:upper_r_2, lower_c_2:upper_c_2]
            for r in range(0, all_indexes.shape[0]):
                lower_r_1 = self.rss + all_indexes[r] * self.lss
                upper_r_1 = lower_r_1 + self.lss
                lower_c_1 = self.rss + all_indexes[r] * self.lss
                upper_c_1 = lower_c_1 + self.lss
                lower_r_2 = self.rss + self.lss * r
                upper_r_2 = lower_r_2 + self.lss
                lower_c_2 = self.rss + self.lss * c
                upper_c_2 = lower_c_2 + self.lss
                omega123_full[lower_r_1:upper_r_1, lower_c_1:upper_c_1] = omega123[lower_r_2:upper_r_2, lower_c_2:upper_c_2]

        sparsified_omega = omega + omega123_full
        sparsified_xi = xi + (sparsified_omega - omega) @ mean

        self.omega.omega_matrix = sparsified_omega
        self.xi.xi_vector = sparsified_xi
        return



    #def seif_correspondence_test(self, omega, xi, mean, mj, mk):
    def seif_correspondence_test(self, mj, mk):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        # breakpoint()
        blanket_b = np.zeros((0, omega.omega_matrix.shape[0] - 3))
        overlap = False
        in_blanket = np.empty(0, dtype=int)
        for i in range(3, omega.omega_matrix.shape[0], 2):
            [found1, found2] = [False, False]
            if not np.allclose(omega.omega_matrix[3 + mj * 2:5 + mj * 2, i:i + 2], np.zeros((2, 2))):
                found1 = True
            if not np.allclose(omega.omega_matrix[3 + mk * 2:5 + mk * 2, i:i + 2], np.zeros((2, 2))):
                found2 = True

            if found1 or found2:
                new_row = np.zeros((2, omega.omega_matrix.shape[0] - 3))
                new_row[:, i - 3:i - 1] = np.identity(2)
                blanket_b = np.concatenate((blanket_b, new_row), axis=0)
                in_blanket = np.append(in_blanket, (i - 3) // 2)

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
                lowest_f = (np.where(open_[:, 4] == np.amin(open_[:, 4])))[0]
                closed_ = np.concatenate((closed_, open_[lowest_f, :]), axis=0)

                open_ = np.delete(open_, lowest_f, 0)

                # Look at all available links to nodes, only do something if a link exists (and it isn't the current node)
                for i in range(3, omega.omega_matrix.shape[0], 2):
                    if (not np.allclose(omega.omega_matrix[3 + int(closed_[-1, 0]) * 2:5 + int(closed_[-1, 0]) * 2, i:i + 2],
                                        np.zeros((2, 2)))) and closed_[-1, 0] != (i - 3) // 2:

                        link_lmark = (i - 3) // 2
                        # if node is already in closed_ then ignore it
                        if len(np.where(closed_[:, 0] == link_lmark)[0]) > 0:
                            continue
                        # if new node is not in open, add it with all its characteristics
                        elif len(np.where(open_[:, 0] == link_lmark)[0]) == 0:
                            x = mean[i, 0]
                            y = mean[i + 1, 0]
                            parent = int(closed_[-1, 0])
                            g = closed_[np.where(closed_[:, 0] == parent), 5][0, 0] + math.sqrt(
                                (x - mean[3 + (parent * 2), :]) ** 2 + (y - mean[4 + (parent * 2), :]) ** 2)
                            h = math.sqrt((x - mean[3 + (mk * 2), :]) ** 2 + (y - mean[4 + (mk * 2), :]) ** 2)
                            f = g + h
                            open_ = np.concatenate((open_, np.array([[float(link_lmark), float(parent), x, y, f, g, h]])),
                                                   axis=0)

                        # else new node is already in open, update it if a better route is available
                        else:
                            # index of item in open_
                            openi = int(np.where(open_[:, 0] == link_lmark)[0])
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
            blanket_b = np.zeros((0, omega.omega_matrix.shape[0] - 3))
            in_blanket = []
            for lmark in path:
                for i in range(3, omega.omega_matrix.shape[0], 2):
                    if not np.allclose(omega.omega_matrix[3 + lmark * 2:5 + lmark * 2, i:2 + i], np.zeros((2, 2))) and (
                            (i - 3) // 2 not in in_blanket):
                        new_row = np.zeros((2, omega.omega_matrix.shape[0] - 3))
                        new_row[:, i - 3:i - 1] = np.identity(2)
                        blanket_b = np.concatenate((blanket_b, new_row), axis=0)
                        in_blanket = np.append(in_blanket, (i - 3) // 2)
            np.savetxt("MarkovBlanket.csv", blanket_b, fmt='%   1.3f', delimiter=",")

        # print(blanket_b)
        # 8

        # 9
        # breakpoint()
        # 10
        local_omega = blanket_b @ omega.omega_matrix[3:, 3:] @ blanket_b.transpose()
        local_xi = blanket_b @ xi.xi_vector[3:, :]

        # print(omega.omega_matrix.round(3))
        # print(local_omega.round(3))
        # print(np.linalg.pinv(local_omega) @ local_xi)

        covB = np.linalg.pinv(local_omega)

        # 11
        meanB = covB @ local_xi
        # 12

        F_delta = np.zeros((2, local_omega.shape[0]))
        # F_delta[:, 0:2] = np.identity(2)
        in_blanket = np.sort(in_blanket)
        jIndex = np.where(in_blanket == mj)[0][0]
        kIndex = np.where(in_blanket == mk)[0][0]
        F_delta[:, jIndex * 2:jIndex * 2 + 2] = np.identity(2)
        F_delta[:, kIndex * 2:kIndex * 2 + 2] = -np.identity(2)

        # 13
        cov_delta = np.linalg.pinv(F_delta @ local_omega @ F_delta.transpose())

        # 14
        mean_delta = cov_delta @ F_delta @ local_xi
        np.savetxt("cov_delta.csv", cov_delta, fmt='%   1.3f', delimiter=",")

        # 15
        gaussian = multivariate_normal.pdf(0, mean_delta[:, 0], cov_delta, True)
        return gaussian


    #def map_correct(self, omega, xi, mean):
    def map_correct(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        l1 = 3
        l2 = 17
        match_probability = self.seif_correspondence_test(omega, xi, mean, l1, l2)
        # breakpoint()
        if match_probability > 0.25:
            F_mjmk = np.zeros((2, omega.omega_matrix.shape[0] - 3))
            F_mjmk[:, 2 * l1:2 + 2 * l1] = np.identity(2)
            F_mjmk[:, 2 * l2:2 + 2 * l2] = -np.identity(2)
            update = F_mjmk.transpose() @ np.diag([1000000, 1000000]) @ F_mjmk
            np.savetxt("update.csv", update, fmt='%   1.3f', delimiter=",")
            omega.omega_matrix[3:, 3:] += update

            Fi1 = np.zeros((2, 3 + 2 * l1))
            Fi2 = np.identity(2)
            Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (l1 + 1)))
            Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)
            Fi_transpose = np.transpose(Fi)
            mean_it = np.linalg.inv(Fi @ omega.omega_matrix @ Fi_transpose) @ Fi @ (
                    xi.xi_vector - (omega.omega_matrix @ mean) + omega.omega_matrix @ Fi_transpose @ Fi @ mean)
            mean[3 + (l1 * 2):3 + (l1 * 2) + 2, :] = mean_it

            Fi1 = np.zeros((2, 3 + 2 * l2))
            Fi2 = np.identity(2)
            Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (l2 + 1)))
            Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)
            Fi_transpose = np.transpose(Fi)
            mean_it = np.linalg.inv(Fi @ omega.omega_matrix @ Fi_transpose) @ Fi @ (
                    xi.xi_vector - (omega.omega_matrix @ mean) + omega.omega_matrix @ Fi_transpose @ Fi @ mean)
            mean[3 + (l2 * 2):3 + (l2 * 2) + 2, :] = mean_it
        return

    def gnss_update(self, z):
        # Get estimate for velocity by gnss
        if z is not None:
            omega = self.omega.omega_matrix
            # xi = self.xi.xi_vector
            mean = self.mean

            h = mean[0:2, :]
            grad_h_x = np.array([[1., 0., 0.],
                                 [0., 1., 0.]])
            grad_h_w = np.array([[1., 0.],
                                 [0., 1.]])

            Fx = np.concatenate((np.identity(3), np.zeros((3, omega.shape[0] - 3))), axis=1)
            cov_x = np.linalg.pinv(Fx @ omega @ Fx.transpose())

            s1 = grad_h_w @ self.gnss_noise_covariance_q @ grad_h_w.transpose()
            s2 = (grad_h_x @ cov_x[0:3, 0:3] @ grad_h_x.transpose())
            s = s1 + s2

            w = cov_x[0:3, 0:3] @ grad_h_x.transpose() @ np.linalg.pinv(s)

            mean[0:3, :] = mean[0:3, :] + (w @ (z - h))
            cov_x[0:3, 0:3] += - w @ s @ w.transpose()
            omega_x = np.linalg.pinv(cov_x)
            # self.omega.omega_matrix[0:3,:] = omega_x[0:3,:]
            # self.omega.omega_matrix[:,0:3] = omega_x[:,0:3]
            omega[0:3, 0:3] = omega_x[0:3, 0:3]

            xi = omega @ mean

            self.omega.omega_matrix = omega
            self.xi.xi_vector = xi
            self.mean = mean
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

        if type(omega) == Omega:
            omega = omega.omega_matrix

        if type(xi) == Xi:
            xi = xi.xi_vector

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
def fast_motion_update(omega, xi, mean, R, Fx, delta_mat, delta):

    Fx_transpose = np.transpose(Fx)
    # 5
    psi = Fx_transpose @ delta_mat @ Fx
    psi_transpose = np.transpose(psi)

    # 6
    lambda_ = (psi_transpose @ omega) + (omega @ psi) + (psi_transpose @ omega @ psi)
    #lambda_ = np.zeros(omega.omega_matrix.shape)
    #lambda_[0:3, :] += delta_mat.transpose() @ omega.omega_matrix[0:3, :]
    #lambda_[:, 0:3] += omega.omega_matrix[:, 0:3] @ delta_mat
    #lambda_[0:3, 0:3] += delta_mat.transpose() @ omega.omega_matrix[0:3, 0:3] @ delta_mat

    # 7
    phi = omega + lambda_

    # 8
    kappa = phi @ Fx_transpose @ (np.linalg.inv(np.linalg.inv(R) + Fx @ phi @ Fx_transpose)) @ Fx @ phi
    #kappa = phi[:, 0:3] @ (np.linalg.inv(np.linalg.inv(self.noise_covariance_r) + phi[0:3, 0:3])) @ phi[0:3, :]

    # 9
    omega = phi - kappa

    # 10
    xi = xi + ((lambda_ - kappa) @ mean) + (omega @ Fx_transpose @ delta)

    # 11
    mean[0:3, :] += delta

    return omega, xi, mean


#@jit(nopython=True)
def fast_motion_update2(omega, xi, mean, R, delta_mat, delta, active):

    Fx_active = np.zeros((3,omega.shape[0]))
    Fx_active[0:3,0:3] = np.identity(3)
    np.sort(active)
    for i in active:
        Fx_active = np.concatenate((Fx_active, np.zeros((2, omega.shape[0]))), axis=0)
        Fx_active[-2:, 2*i+3:2*i+5] = np.identity(2)

    omega_active = Fx_active @ omega @ Fx_active.transpose()

    Fx = np.concatenate((np.identity(3), np.zeros((3, omega_active.shape[0]-3))), axis=1)
    Fx_transpose = Fx.transpose()
    # 5
    psi = Fx_transpose @ delta_mat @ Fx
    psi_transpose = np.transpose(psi)

    # 6
    lambda_ = (psi_transpose @ omega_active) + (omega_active @ psi) + (psi_transpose @ omega_active @ psi)

    # 7
    phi = omega_active + lambda_

    # 8
    kappa = phi @ Fx_transpose @ (np.linalg.inv(np.linalg.inv(R) + Fx @ phi @ Fx_transpose)) @ Fx @ phi
    # kappa = phi[:, 0:3] @ (np.linalg.inv(np.linalg.inv(self.noise_covariance_r) + phi[0:3, 0:3])) @ phi[0:3, :]

    # 9
    omega_active = phi - kappa

    # 10
    xi_active = (Fx_active @ xi) + ((lambda_ - kappa) @ (Fx_active @ mean)) + (omega_active @ Fx_transpose @ delta)

    omega[0:3, 0:3] = omega_active[0:3, 0:3]
    xi[0:3, :] = xi_active[0:3, :]

    for c in range(0, active.shape[0]):
        omega[0:3, active[c] * 2 + 3:active[c] * 2 + 5] = omega_active[0:3, 2 * c + 3:2 * c + 5]
        omega[active[c] * 2 + 3:active[c] * 2 + 5, 0:3] = omega_active[2 * c + 3:2 * c + 5, 0:3]
        xi[active[c] * 2 + 3:active[c] * 2 + 5, :] = xi_active[c * 2 + 3:c * 2 + 5, :]
        for r in range(0, active.shape[0]):
            omega[active[r] * 2 + 3:active[r] * 2 + 5, active[c] * 2 + 3:active[c] * 2 + 5] = omega_active[2 * r + 3:2 * r + 5, 2 * c + 3:2 * c + 5]

    # 11
    mean[0:3, :] += delta
    return omega, xi, mean


@jit(nopython=True)
def fast_sparse(omega, xi, mean, active, to_deactivate):
    omega_size = omega.shape

    F_x_act_deact = np.zeros(omega_size)
    F_x_act_deact[0:3, 0:3] = np.identity(3)

    F_deact = np.zeros(omega_size)

    F_x_deact = np.zeros(omega_size)
    F_x_deact[0:3, 0:3] = np.identity(3)

    F_x = np.zeros(omega_size)
    F_x[0:3, 0:3] = np.identity(3)

    for i in active:
        F_x_act_deact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    for i in to_deactivate:
        F_x_act_deact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)
        F_x_deact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)
        F_deact[3 + (i * 2):5 + (i * 2), 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    omega_0t = F_x_act_deact @ omega @ F_x_act_deact
    omega1 = -(omega_0t @ F_deact @ np.linalg.pinv(F_deact @ omega_0t @ F_deact) @ F_deact @ omega_0t)
    omega2 = omega_0t @ F_x_deact @ np.linalg.pinv(F_x_deact @ omega_0t @ F_x_deact) @ F_x_deact @ omega_0t
    omega3 = -(omega @ F_x @ np.linalg.pinv(F_x @ omega @ F_x) @ F_x @ omega)
    sparsified_omega = omega + omega1 + omega2 + omega3
    sparsified_xi = xi + (sparsified_omega - omega) @ mean
    return sparsified_omega, sparsified_xi


#@jit(nopython=True)
def fast_sparse2(omega, xi, mean, active, to_deactivate):
    all_indexes = np.sort((np.concatenate((active, to_deactivate), axis=0)).astype(np.int32))
    np.sort(active)
    np.sort(to_deactivate)
    omega_size = omega.shape[0]
    omega0_size = all_indexes.shape[0] * 2 + 3

    F_x_act_deact = np.zeros((3, omega_size))
    F_x_act_deact[0:3, 0:3] = np.identity(3)

    F_deact = np.zeros((3, omega0_size))

    F_x_deact = np.zeros((3, omega0_size))
    F_x_deact[0:3, 0:3] = np.identity(3)

    F_x = np.zeros((3, omega0_size))
    F_x[0:3, 0:3] = np.identity(3)
    for i in all_indexes:
        F_x_act_deact = np.concatenate((F_x_act_deact, np.zeros((2, omega_size))), axis=0)
        F_x_act_deact[-2:, 3 + (i * 2):5 + (i * 2)] = np.identity(2)

    if to_deactivate.shape[0] != 0:
        for i in to_deactivate:
            pos = np.where(all_indexes == i)[0][0]

            F_x_deact = np.concatenate((F_x_deact, np.zeros((2, omega0_size))), axis=0)
            F_x_deact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

            F_deact = np.concatenate((F_deact, np.zeros((2, omega0_size))), axis=0)
            F_deact[-2:, 3 + (pos * 2):5 + (pos * 2)] = np.identity(2)

    omega_0t = F_x_act_deact @ omega @ F_x_act_deact.transpose()
    omega1 = -(omega_0t @ F_deact.transpose() @ np.linalg.pinv(F_deact @ omega_0t @ F_deact.transpose()) @ F_deact @ omega_0t)
    omega2 = omega_0t @ F_x_deact.transpose() @ np.linalg.pinv(F_x_deact @ omega_0t @ F_x_deact.transpose()) @ F_x_deact @ omega_0t
    omega3 = -(omega_0t @ F_x.transpose() @ np.linalg.pinv(F_x @ omega_0t @ F_x.transpose()) @ F_x @ omega_0t)
    omega123 = omega1 + omega2 + omega3

    omega123_full = np.zeros((omega_size, omega_size))
    omega123_full[0:3, 0:3] = omega123[0:3, 0:3]

    for c in range(0, all_indexes.shape[0]):
        omega123_full[0:3, all_indexes[c] * 2 + 3:all_indexes[c] * 2 + 5] = omega123[0:3, 2 * c + 3:2 * c + 5]
        omega123_full[all_indexes[c] * 2 + 3:all_indexes[c] * 2 + 5, 0:3] = omega123[2 * c + 3:2 * c + 5, 0:3]
        for r in range(0, all_indexes.shape[0]):
            omega123_full[all_indexes[r] * 2 + 3:all_indexes[r] * 2 + 5, all_indexes[r] * 2 + 3:all_indexes[r] * 2 + 5] = omega123[2 * r + 3:2 * r + 5, 2 * c + 3:2 * c + 5]

    sparsified_omega = omega + omega123_full
    sparsified_xi = xi + (sparsified_omega - omega) @ mean
    return sparsified_omega, sparsified_xi

#@jit(nopython=True)
def fast_jacobianh_and_updates(omega, xi, mean, Q, delta, q, i, z, z_hat):
    h1 = np.array([[-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0], 0],
                   [delta[1, 0], -delta[0, 0], -q]])

    h2 = np.zeros((2, 2 * i))

    h3 = np.array([[math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0]],
                   [-delta[1, 0], delta[0, 0]]])

    h4 = np.zeros((2, (mean.shape[0] - 3) - 2 * (i + 1)))

    h_it = 1 / q * np.concatenate((h1, h2, h3, h4), axis=1)

    xi_update = np.transpose(h_it) @ np.linalg.inv(Q) @ (z - z_hat + (h_it @ mean))

    # resize array with new columns and rows as zeros
    if xi_update.shape > xi.shape:
        new_array = np.zeros(xi_update.shape)
        new_array[0:xi.shape[0], 0:xi.shape[1]] = xi
        xi = new_array

    omega_update = np.transpose(h_it) @ np.linalg.inv(Q) @ h_it
    if omega_update.shape > omega.shape:
        new_matrix = np.zeros(omega_update.shape)
        new_matrix[0:omega.shape[0], 0:omega.shape[0]] = omega
        omega = new_matrix

    xi += xi_update
    omega += omega_update
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
    updated_mean = np.linalg.inv(Fx @ omega @ Fx.transpose()) @ Fx @ (xi - (omega @ mean) + (omega @ Fx.transpose() @ Fx @ mean))
    mean[0:3, :] = updated_mean
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
    # self.omega.omega_matrix[0:3,:] = omega_x[0:3,:]
    # self.omega.omega_matrix[:,0:3] = omega_x[:,0:3]
    omega[0:3, 0:3] = omega_x[0:3, 0:3]

    xi = omega @ mean
    return omega, xi, mean
'''
