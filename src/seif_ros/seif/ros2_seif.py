#!/usr/bin/env python
"""
Online GraphSLAM (SEIF) based on Probabilistic Robotics (2004), Thrun et al

"""
import sys
import time

import numpy as np
import math
import matplotlib.pyplot as plt
import rclpy
from collections import deque

from ekf.omega import *
from scipy.stats import multivariate_normal
from numba import jit

from helpers.listener import BaseListener
from obr_msgs.msg import CarPos, State, ConeArray, Cone  # WheelSpeeds - msg type does not exist
from geometry_msgs.msg import Point, Twist
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates
from rclpy.qos import qos_profile_sensor_data


class Listener(BaseListener):
    def __init__(self, sparsityN):
        super().__init__('slam')

        # =====SEIF SPECIFIC INIT=====
        self.omega = Omega2()
        self.xi = Xi()
        self.mean = np.zeros((3, 1))

        self.motionNoiseCovariance = [1. ** 2, 1. ** 2, 1. ** 2]
        self.measureNoiseCovariance = [1. ** 2, 10. ** 2]
        self.gnssNoiseCovariance = [0.1 ** 2, 0.1 ** 2]
        self.noiseCovarianceR = np.diag(self.motionNoiseCovariance)
        self.noiseCovarianceQ = np.diag(self.measureNoiseCovariance)
        self.gnssNoiseCovarianceQ = np.diag(self.gnssNoiseCovariance)

        self.sparsityN = sparsityN
        self.active = deque([], maxlen=sparsityN)
        self.toDeactivate = deque([], maxlen=sparsityN)

        # fast_motion_update2(np.zeros((3,3)), np.zeros((3,1)), np.zeros((3,1)), np.identity(3), np.zeros((3,3)), np.zeros((3,1)), np.asarray(self.active))
        # fast_sparse2(self.omega.omegaMatrix, self.xi.xiVector, self.mean, np.asarray(self.active), np.asarray(self.toDeactivate), True)
        # =====SYSTEM=====
        # @Q_IMU: Noise for linear acc and angular velocity : should be linear and angular velocities
        self.Q_IMU = np.array([[50, 0],
                               [0, 10]])
        # @R_GNSS: Measurement noise for latitude and longitude
        self.R_GNSS = np.array([[0.01, 0],
                                [0, 0.01]])
        # @centerLat: GNSS latitude of 0, 0.
        # @centerLong: GNSS longitude of 0, 0.
        self.centerLat = 0
        self.centerLong = 0

        # @R_SPEEDOMETER: Measurement noise for speed
        self.R_SPEEDOMETER = np.array([[0.1]])

        # Time trackers
        # @ws_time: wheelspeed time, helps us determine most recent data
        # @gnss_time: GPS time
        # @imu_time: IMU time
        # @last_update_time: records time of last update
        self.start_time = 0
        self.ws_time = 0
        self.gnss_time = 0
        self.gnss_time_prev = 0
        self.imu_time = 0
        self.last_update_time = 0

        # @ws: becomes the most recent message from wheelspeed
        # @gnss: becomes the most recent message from the GPS
        # @imu: becomes the most recent message from IMU
        self.ws = None
        self.wsPoint1 = np.array([[0.], [0.]])
        self.wsPoint2 = np.array([[0.], [0.]])
        self.gnss = None
        self.imu = None
        self.cones = None
        self.linearVel = 0.0

        # @pub: Publisher for the car pose (x,y,theta)
        self.pub = self.create_publisher(CarPos, '/car/position', 10)

        # Storing the values retrieved directly from sensors for plotting
        self.position_truth = None
        self.trajectory_truth = np.zeros((2, 0))
        self.position_by_IMU = [[0, 0]]  # Stores latest calculated x, y vector
        self.trajectory_by_IMU = np.zeros((2, 0))  # Stores concatenation of all previously calculated x,y vectors
        self.position_by_GNSS = None
        self.trajectory_by_GNSS = np.zeros((2, 0))
        self.position_by_WSS = [[0, 0]]
        self.trajectory_by_WSS = np.zeros((2, 0))
        self.position_by_SEIF = [[0, 0]]
        self.trajectory_by_SEIF = np.zeros((2, 0))

        # @gnss_sub: creates subscription to the gnss
        # @wheel_speed_sub: creates subscription to the wheelspeed (NOT CURRENTLY PUBLISHED TO)
        # @imu_sub: listens to the IMU data
        # Problem with listening to best_effort reliability QoS messages: https://answers.ros.org/question/332207/cant-receive-data-in-python-node/
        # requires including qos_profile_sensor_data
        # @models_sub: Subsrriber for the pose of a 3D module of the car to be used as a ground truth
        self.gnss_sub = self.create_subscription(NavSatFix, '/peak_gps/gps/truth', self.gnss_callback,
                                                 qos_profile_sensor_data)  # /oxts/fix '/peak_gps/gps/truth'
        self.wheel_speed_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.ws_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/peak_gps/imu/truth', self.imu_callback,
                                                qos_profile_sensor_data)  # /oxts/imu '/peak_gps/imu/truth' 10
        self.models_sub = self.create_subscription(ModelStates, '/gazebo/model_states', self.gt_pose_callback, 10)
        self.links_sub = self.create_subscription(LinkStates, '/gazebo/link_states', self.gt_links_callback, 10)
        self.cone_detect_sub = self.create_subscription(ConeArray, '/cones/positions', self.cone_detect_callback, 10)
        # self.control_sub = self.create_subscription(Twist,'/gazebo/cmd_vel', self.control_callback, 10)

        # Helpers to compute positions relative to the origin of the coordinate system
        self.model_start_collected = False
        self.gt_origin = None
        self.map_collected = False
        self.gt_cones = np.zeros((2, 0))

        # @timer: timer for scheduling the SEIF vehicle state estimation calculation
        self.timer = self.create_timer(0.1, self.timer_callback)

        # History (matrix) of the latitude-longitude values from GPS for plotting
        self.posHist = np.zeros((2, 0))  # latitude longitude
        self.counter = 0

        # =====PLOTTING=====
        self.coneplt = None
        return

    # =====SEIF SPECIFIC METHODS=====
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

        # while delta[2, :] > math.pi:
        #    delta[2, :] -= 2 * math.pi
        # while delta[2, :] < -math.pi:
        #    delta[2, :] += 2 * math.pi

        # 4 #ERRATA says this should be negative
        deltaMat = np.array(
            [[0, 0, (v / w) * math.cos(self.mean[2, 0]) - (v / w) * math.cos(self.mean[2, 0] + w * timeStep)],
             [0, 0, (v / w) * math.sin(self.mean[2, 0]) - (v / w) * math.sin(self.mean[2, 0] + w * timeStep)],
             [0, 0, 0]])

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
                if (mu_j[0, 0] - mean[3 + 2 * n, 0]) ** 2 + (mu_j[1, 0] - mean[4 + 2 * n, 0]) ** 2 < 0.25:
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
                    del (self.active[0])
                if i in self.toDeactivate:
                    del (self.toDeactivate[list(self.toDeactivate).index(i)])
                self.active.append(i)

            # 8
            delta = mu_j - mean[0:2, :]

            # 9
            q = (np.transpose(delta) @ delta)[0][0]

            # 10
            zhat = np.array([[math.sqrt(q)],
                             [math.atan2(delta[1, 0], delta[0, 0]) - mean[2, 0]]])

            # bring angle back into -pi < theta < pi range
            # while zhat[1, :] > math.pi:
            #    zhat[1, :] -= 2 * math.pi
            # while zhat[1, :] < -math.pi:
            #    zhat[1, :] += 2 * math.pi
            zhat[1, :] = (zhat[1, :] + np.pi) % (2 * np.pi) - np.pi

            omega.omegaMatrix, xi.xiVector = fast_jacobianh_and_updates(omega.omegaMatrix, xi.xiVector, mean,
                                                                        self.noiseCovarianceQ, delta, q, i, z, zhat)

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
        if len(self.active) == 0:
            return

        self.mean = fast_state_estimation(omega.omegaMatrix, xi.xiVector, mean, np.asarray(self.active))
        return

    # def seif_sparsification(self, xi, omega, mean):
    def seif_sparsification(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        self.omega.omegaMatrix, self.xi.xiVector = fast_sparse2(omega.omegaMatrix, xi.xiVector, mean,
                                                                np.asarray(self.active), np.asarray(self.toDeactivate),
                                                                False)
        return

    def seif_correspondence_test(self, mj, mk):
        xi = self.xi
        omega = self.omega
        mean = self.mean

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
                lowestF = (np.where(open_[:, 4] == np.amin(open_[:, 4])))[0]
                closed_ = np.concatenate((closed_, open_[lowestF, :]), axis=0)

                open_ = np.delete(open_, lowestF, 0)

                # Look at all available links to nodes, only do something if a link exists (and it isn't the current node)
                for i in range(3, omega.omegaMatrix.shape[0], 2):
                    if (
                    not np.allclose(omega.omegaMatrix[3 + int(closed_[-1, 0]) * 2:5 + int(closed_[-1, 0]) * 2, i:i + 2],
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
                            open_ = np.concatenate(
                                (open_, np.array([[float(linkLmark), float(parent), x, y, f, g, h]])),
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

        # 8

        # 9
        # 10
        localOmega = blanketB @ omega.omegaMatrix[3:, 3:] @ blanketB.transpose()
        localXi = blanketB @ xi.xiVector[3:, :]

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

    def map_correct(self):
        xi = self.xi
        omega = self.omega
        mean = self.mean

        l1 = 3
        l2 = 17
        matchProbability = self.seif_correspondence_test(omega, xi, mean, l1, l2)

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

    # =====MAIN CALLBACK=====
    '''
    Function: timer_callback()
    Purpose: To trigger EKF. Checks to see if variables used are from most
             recent updates, then runs publisher functions.
    Parameters: @self, to access member data
    Outputs: no returns, prints information and publishes position to messages
    '''

    def timer_callback(self):
        t0 = time.time()
        if self.start_time == 0:
            self.start_time = time.time()

        # print("Running")
        if (self.last_update_time != 0):
            dt = self.imu_time - self.last_update_time
        else:
            dt = 0.01

        t1 = time.time()
        # print("v:",self.linearVel)
        if self.imu is not None:
            # print("making movement update")
            self.linearVel += self.imu.linear_acceleration.x * dt
            self.seif_motion_update((self.linearVel, self.imu.angular_velocity.z), dt)
            self.last_update_time = self.imu_time
            # print("IMU pos:", self.mean[0:3].transpose().round(3))
        t2 = time.time()
        if self.position_by_GNSS is not None:
            # print(np.array([[self.gnss.longitude], [self.gnss.latitude]]))
            self.gnss_update(np.array([[self.gnss.longitude], [self.gnss.latitude]]))
        t3 = time.time()
        if (self.cones is not None):
            # print("cones:\n",self.cones)
            self.seif_measurement_update(self.cones)
        t4 = time.time()
        self.seif_update_state_estimation()
        t5 = time.time()
        self.seif_sparsification()
        t6 = time.time()
        # print("x:"+str(self.mean[0,0].round(3))+" y:"+str(self.mean[1,0].round(3))+" a:"+str(self.mean[2,0].round(3))+" v:"+str(self.linearVel))
        if (time.time() - self.start_time) > 30:
            np.savetxt("mean.csv", self.mean, fmt='%   1.3f', delimiter=",")
            self.start_time = time.time()
            print("saved mean.csv")

        # Prints most recent position
        np.set_printoptions(suppress=True)
        # print("\n\n\n",self.mean.round(3))
        self.position_by_SEIF = self.mean[0:2, :]
        self.trajectory_by_SEIF = np.hstack((self.trajectory_by_SEIF, self.position_by_SEIF))
        self.plot_vehicle_state(plot_true=True, plot_IMU=False, plot_GNSS=False, plot_SEIF=True, plot_cones=False)
        t7 = time.time()
        # Publishes car position message to topic
        self.pub.publish(
            CarPos(position=Point(x=float(self.mean[0, 0]), y=float(self.mean[1, 0])), angle=float(self.mean[2, 0])))
        # self.plot_pub.publish(Point(x=float(self.mean[0,0]), y=float(self.mean[1, 0])))
        print("td1=", t1 - t0, " td2=", t2 - t1, " td3=", t3 - t2, " td4=", t4 - t3, " td5=", t5 - t4, " td6=", t6 - t5,
              " td7=", t7 - t6, " tot=", t7 - t0)

    # =====CALLBACK METHODS=====
    # wheel speeds
    def ws_callback(self, msg):
        self.ws_time = time.time()
        self.ws = msg

    # GPS
    def gnss_callback(self, msg: NavSatFix):
        if (msg.latitude != "nan" and msg.latitude != np.nan) and (
                msg.longitude != "nan" and msg.longitude != np.nan):
            # self.gnss_time_prev = self.gnss_time
            self.gnss_time = time.time()
            self.gnss = msg
            # latKilos is the distance a single degree of latitude covers in kilometers.
            # longKilos is the distance a single degree of longitude covers in kilometers.
            self.latKilos = 111.3195
            self.longKilos = (self.latKilos * math.cos(self.gnss.latitude)) * 1000

            if self.centerLat == .0 and self.centerLong == .0:
                self.centerLat = self.gnss.latitude * (self.latKilos * 1000)
                self.centerLong = self.gnss.longitude * self.longKilos
                # Account for the position of the GPS relative to the car's center.
                self.centerLong -= 1.215 * math.cos(self.mean[2, 0])
                self.centerLat -= 1.215 * math.sin(self.mean[2, 0])
                self.gnss.latitude = .0
                self.gnss.longitude = .0
            else:
                self.gnss.latitude = self.centerLat - (self.gnss.latitude * (self.latKilos * 1000))
                self.gnss.longitude = self.centerLong - (self.gnss.longitude * self.longKilos)

            # print(self.gnss.latitude, self.gnss.longitude)
            self.position_by_GNSS = np.array([[self.gnss.longitude], [self.gnss.latitude]])
            self.trajectory_by_GNSS = np.hstack(
                (self.trajectory_by_GNSS, np.array([[self.gnss.longitude], [self.gnss.latitude]])))

            # self.gnss_update([self.gnss.longitude, self.gnss.latitude])
            # self.gnss_prev = [self.gnss.longitude, self.gnss.latitude]

    # IMU
    def imu_callback(self, msg: Imu):
        # print("Received IMU")
        self.imu_time = time.time()
        self.imu = msg

    def control_callback(self, msg: Twist):
        str(msg)  # For some reason this is needed to access msg.linear.x
        # self.linearVel = self.linearVel - (self.linearVel-msg.linear.x)*0.01
        # self.angularVel = self.angularVel - (self.angularVel-msg.angular.z)*0.01
        # self.u = np.array([self.v, self.theta]).reshape(2, 1)

    # CONE DETECTION
    def cone_detect_callback(self, msg: ConeArray):
        self.cones = np.empty((0, 3))
        # print(msg)
        if msg is not None:
            for c in msg.cones:
                # print(bearing)
                # RETURN TYPE 1: (relative x, relative y, colour)
                # correct angle offset of measurements (for some reason they're rotated by pi/2)
                x = c.position.x * math.cos(self.mean[2, 0] - math.pi / 2) + c.position.y * -math.sin(
                    self.mean[2, 0] - math.pi / 2)
                y = c.position.x * math.sin(self.mean[2, 0] - math.pi / 2) + c.position.y * math.cos(
                    self.mean[2, 0] - math.pi / 2)
                data = np.array([[x, y, c.label.label]])

                # RETURN TYPE 2: (range, bearing, colour) NOTE: requires x,y from method 1
                # print(dx, " , ", dy)
                range_ = math.sqrt((x) ** 2 + (y) ** 2)
                bearing = math.atan2(y, x) - self.mean[2, 0]
                # while bearing > math.pi:
                #    bearing -= 2 * math.pi
                # while bearing < -math.pi:
                #    bearing += 2 * math.pi
                bearing = (bearing + np.pi) % (2 * np.pi) - np.pi

                data = np.array([[range_, bearing, c.label.label]])

                self.cones = np.concatenate((self.cones, data), axis=0)

        # print(self.cones)

    def gt_pose_callback(self, msg: ModelStates):
        if not self.model_start_collected:
            self.model_start_collected = True
            self.gt_origin = np.array([[msg.pose[2].position.x], [msg.pose[2].position.y]])
        self.position_truth = np.array([[msg.pose[2].position.x], [msg.pose[2].position.y]]) - self.gt_origin
        self.trajectory_truth = np.hstack((self.trajectory_truth, self.position_truth))

    def gt_links_callback(self, msg: LinkStates):
        # print("Got cones callback")
        if not self.map_collected:
            if self.model_start_collected and len(msg.pose) > 0:
                self.map_collected = True

                rear_wheel_position = None
                front_wheel_position = None
                for i in range(len(msg.pose)):
                    pose = msg.pose[i]
                    if "cone" in msg.name[i]:
                        cone_pos = np.array([[pose.position.x], [pose.position.y]]) - self.gt_origin
                        self.gt_cones = np.hstack((self.gt_cones, cone_pos))
                    elif "rear_right_wheel" in msg.name[i]:
                        rear_wheel_position = np.array([pose.position.x, pose.position.y])
                        print(rear_wheel_position)
                    elif "front_right_wheel" in msg.name[i]:
                        front_wheel_position = np.array([pose.position.x, pose.position.y])
                        print(front_wheel_position)

                # Calculate the initial heading (orientation) of the vehicle
                assert rear_wheel_position is not None and front_wheel_position is not None, "Couldn't collect wheel positions"
                v = front_wheel_position - rear_wheel_position
                theta = np.arctan2(v[1], v[0])
                self.mean[2, 0] = theta
                print("SET THE ORIENTATION TO {}".format(theta))

    # =====PLOTTING METHOD=====
    '''
    Function: plot_vehicle_state()
    Purpose: Plots the vehicle state
    Paramaters:
        @plot_...: boolean values that control what trajectory to plot
    Outputs:
        null
    '''

    def plot_vehicle_state(self, plot_true=False, plot_IMU=False, plot_GNSS=False, plot_SEIF=False, plot_cones=True):
        plt.cla()  # Clear current plot

        # Stops the visualization with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event', lambda event:
            [exit(0) if event.key == 'escape' else None])

        if plot_true and self.position_truth is not None:
            x = self.trajectory_truth[0, :]
            y = self.trajectory_truth[1, :]
            # Plot xEst with solid red line (see this link for styling: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html)
            plt.plot(x, y, "-k", label="True Path")
        if plot_IMU and self.position_by_IMU is not None:
            x = self.trajectory_by_IMU[0, :]
            y = self.trajectory_by_IMU[1, :]
            plt.plot(x, y, "-b", label="Path by IMU")
        if plot_GNSS and self.position_by_GNSS is not None:
            x = self.trajectory_by_GNSS[0, :]
            y = self.trajectory_by_GNSS[1, :]
            plt.plot(x, y, "-r", label="Path by GNSS")
        if plot_SEIF:
            x = self.trajectory_by_SEIF[0, :]
            y = self.trajectory_by_SEIF[1, :]
            plt.plot(x, y, "-g", label="Path by SEIF")
        # if plot_cones and self.mean is not None:
        if plot_cones and self.mean is not None:
            if self.coneplt is not None:
                self.coneplt.remove()
            x = np.zeros((len(self.active), 1))
            y = np.zeros((len(self.active), 1))
            c = 0
            for i in self.active:
                x[c, 0] = self.mean[i * 2 + 3, 0]
                y[c, 0] = self.mean[i * 2 + 4, 0]
                c += 1
            self.coneplt = plt.scatter(x, y, marker="x", color="r", label="SEIF Cones")

        # Plot cones
        if self.map_collected:
            plt.plot(self.gt_cones[0, :], self.gt_cones[1, :], ".g", label='True Cones')

        plt.legend()
        plt.title('Visualizing Vehicle State')
        plt.xlabel('X distance (m)')
        plt.ylabel('Y distance (m)')
        plt.axis("equal")
        plt.grid(True)
        plt.pause(0.001)

    # =====HELPER FUNCTIONS=====
    '''
    def prepare_speedometer_update(self) -> tuple:
        h = np.array([[self.pos[0, 3]]])
        grad_h_x = np.array([[0, 0, 0, 1]])
        grad_h_w = np.array([[1]])
        return h, grad_h_x, grad_h_w
    '''

    def gnss_update(self, z):
        # Get estimate for velocity by gnss
        if self.gnss is not None:
            self.omega.omegaMatrix, self.xi.xiVector, self.mean = fast_gnss_update(self.omega.omegaMatrix,
                                                                                   self.xi.xiVector, self.mean, z,
                                                                                   self.gnssNoiseCovarianceQ)

            '''
            delta = z - self.mean[0:2, :]
            q = (np.transpose(delta) @ delta)[0][0]
            zhat = z

            h1 = np.array([
                [-math.sqrt(q) * delta[0, 0], -math.sqrt(q) * delta[1, 0]],
                [delta[1, 0], -delta[0, 0]],
                [math.sqrt(q) * delta[0, 0], math.sqrt(q) * delta[1, 0]],
                [-delta[1, 0], delta[0, 0]]
                ])


            h2 = np.zeros((4, self.omega.omegaMatrix.shape[0]-2))

            h_it = 1 / q * np.concatenate((h1, h2), axis=1)
            print(h_it.shape)
            print("\ndelta:",delta) 
            print("q:",q)
            print("zhat:",z)
            a = (z - zhat + (h_it @ self.mean))
            xiUpdate = np.transpose(h_it) @ np.linalg.inv(self.gnssNoiseCovarianceQ) @ (z - zhat + (h_it @ self.mean))
            omegaUpdate = np.transpose(h_it) @ np.linalg.inv(self.gnssNoiseCovarianceQ) @ h_it

            self.xi.xiVector += xiUpdate
            self.omega.omegaMatrix += omegaUpdate
            Fx = np.concatenate((np.identity(3), np.zeros((3,self.omega.omegaMatrix.shape[0]-3))), axis=1)
            newMean = np.linalg.inv(Fx @ self.omega.omegaMatrix @ Fx.transpose()) @ Fx @ (self.xi.xiVector - (self.omega.omegaMatrix @ self.mean) + (self.omega.omegaMatrix @ Fx.transpose() @ Fx @ self.mean))

            self.mean[0:3,:] = newMean
            self.linearVel = 0.0001*self.linearVel

            #t = self.gnss_time - self.last_update_time
            #d = math.sqrt((newMean[0,0] - self.mean[0,0])**2 + (newMean[] - self.gnss_prev[1])**2)
            #v = d/t
            #if abs(self.linearVel-v) < 10:
            #    self.linearVel = v
            #self.linearVel += (((self.linearVel + v)/2)-self.linearVel)*0.1 # in update format but doesn't work great
            #self.linearVel = ((self.linearVel + v)/2)*0.1 # works very well
            #self.linearVel = ((self.linearVel + v)/2)  # works but badly
            #self.mean[0:2,:] = np.array([[z[0]], [z[1]]])
            #print(self.mean)
            #self.omega.omegaMatrix[0:3,0:3] = np.diag([1/d,1/d,self.omega.omegaMatrix[2,2]])
            #self.xi.xiVector[0:3,:] = self.omega.omegaMatrix[0:3,0:3] @ self.mean[0:3,:]
            #print("GPS updated")
            '''
        return

    def calculate_p_hat(self, w: np.array, s: np.array) -> np.array:
        # p_hat = p - w*s*w'
        p1 = w.transpose()
        p1 = np.matmul(s, p1)
        p1 = np.matmul(w, p1)
        p_hat = self.p - p1
        return p_hat

    def calculate_s_speedometer(self, grad_h_x: np.array, grad_h_w: np.array) -> np.array:
        return self.calculate_s(grad_h_x, grad_h_w, self.R_SPEEDOMETER)

    def calculate_s_gnss(self, grad_h_x: np.array, grad_h_w: np.array) -> np.array:
        return self.calculate_s(grad_h_x, grad_h_w, self.R_GNSS)

    def calculate_s(self, grad_h_x: np.array, grad_h_w: np.array, R: np.array) -> np.array:
        # S = grad_h_w * R_GNSS * grad_h_w' + grad_h_x*P*grad_h_x'
        s1 = grad_h_w.transpose()
        s1 = R * s1
        s1 = np.matmul(grad_h_w, s1)

        s2 = grad_h_x.transpose()
        s2 = np.matmul(self.p, s2)
        s2 = np.matmul(grad_h_x, s2)

        s = s1 + s2
        # print(s)
        return s

    def calculate_x_hat(self, w: np.array, z: np.array, h: np.array) -> np.array:
        # x_hat = x+(W*(z-h)')'
        x1 = z - h
        x1 = x1.transpose()
        x1 = np.matmul(w, x1)
        x1 = x1.transpose()

        x_hat = self.pos + x1
        return x_hat

    def calculate_w(self, grad_h_x: np.array, s: np.array) -> np.array:
        # W = P*grad_h_x'*inv(S)
        w1 = np.inv(s)

        w2 = grad_h_x.transpose()
        w2 = np.matmul(self.p, w2)

        w = np.matmul(w2, w1)
        return w


# =====NUMBA METHODS=====
# numba methods used to dramatically improve runtime of slow numpy matrix operations
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


# @jit(nopython=True)
def fast_motion_update2(omega, xi, mean, R, deltaMat, delta, active):
    Fx_active = np.zeros((3, omega.shape[0]))
    Fx_active[0:3, 0:3] = np.identity(3)
    np.sort(active)
    for i in active:
        Fx_active = np.concatenate((Fx_active, np.zeros((2, omega.shape[0]))), axis=0)
        Fx_active[-2:, 2 * i + 3:2 * i + 5] = np.identity(2)

    omegaActive = Fx_active @ omega @ Fx_active.transpose()

    Fx = np.concatenate((np.identity(3), np.zeros((3, omegaActive.shape[0] - 3))), axis=1)
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
            omega[active[r] * 2 + 3:active[r] * 2 + 5, active[c] * 2 + 3:active[c] * 2 + 5] = omegaActive[
                                                                                              2 * r + 3:2 * r + 5,
                                                                                              2 * c + 3:2 * c + 5]

    # 11
    mean[0:3, :] += delta
    return omega, xi, mean


# @jit(nopython=True)
def fast_sparse(omega, xi, mean, active, toDeactivate, init):
    if not init:
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
    else:
        return


# @jit(nopython=True)
def fast_sparse2(omega, xi, mean, active, toDeactivate, init):
    if not init:
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
        omega1 = -(omega_0t @ F_deact.transpose() @ np.linalg.pinv(
            F_deact @ omega_0t @ F_deact.transpose()) @ F_deact @ omega_0t)
        omega2 = omega_0t @ F_xDeact.transpose() @ np.linalg.pinv(
            F_xDeact @ omega_0t @ F_xDeact.transpose()) @ F_xDeact @ omega_0t
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
        return sparsifiedOmega, sparsifiedXi
    else:
        return


# @jit(nopython=True)
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


# @jit(nopython=True)
def fast_state_estimation(omega, xi, mean, active):
    # 2
    # for small number of active map features

    for i in active:
        Fi1 = np.zeros((2, 3 + 2 * i))
        Fi2 = np.identity(2)
        Fi3 = np.zeros((2, (mean.shape[0] - 3) - 2 * (i + 1)))
        Fi = np.concatenate((Fi1, Fi2, Fi3), axis=1)

        mean_it = np.linalg.inv(Fi @ omega @ Fi.transpose()) @ Fi @ (
                    xi - (omega @ mean) + omega @ Fi.transpose() @ Fi @ mean)
        mean[3 + (i * 2):3 + (i * 2) + 2, :] = mean_it

    # 9
    Fx = np.concatenate((np.identity(3), np.zeros((3, mean.shape[0] - 3))), axis=1)

    # 10
    updatedMean = np.linalg.inv(Fx @ omega @ Fx.transpose()) @ Fx @ (
                xi - (omega @ mean) + (omega @ Fx.transpose() @ Fx @ mean))
    mean[0:3, :] = updatedMean
    return mean


# @jit(nopython=True)
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


def main(args=None):
    rclpy.init(args=args)

    node = Listener(10)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv)
