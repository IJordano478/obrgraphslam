from src.simulation.Track import Track, Cone
from src.simulation.variables import *
import random
import time
import math
import numpy as np


class SimWorld:
    def __init__(self, t, pos: np.array):
        self._track = t
        self._pos = pos
        self._s1 = 0
        self._s2 = 0
        self._s1Command = 0
        self._s2Command = 0
        self.waitTime = 0.05
        self._cone_visibility = t.landmarks

    def sim_get_pos(self):
        return self._pos

    def sim_step(self):
        x = self._pos[0,2]
        y = self._pos[1,2]

        s1 = self._s1 * self.waitTime
        s2 = self._s2 * self.waitTime
        #a = (s2-s1) / (np.exp(4.44265125649))
        a = (s2 - s1) / 0.0001
        if abs(a) > 0.0001:
            rad = (s2+s1) / (2 * a)
            x = rad * math.sin(a)
            y = -rad * (math.cos(a) - 1)
        else:
            x = s1
            y = 0
        d1 = self._s1 - self._s1Command
        d2 = self._s2 - self._s2Command
        accel = d1*d1+d2*d2
        # multiplicative noise on x and a, scaled by acceleration, to simulate slippage
        x = x + x * np.random.normal(0, min(0.5, 0.0001 * accel))
        a = a + a * np.random.normal(0, min(0.5, 0.0001 * accel))

        f = np.array([[math.cos(a), -math.sin(a), x], [math.sin(a), math.cos(a), y], [0., 0.,1.]])
        self._pos = self._pos.dot(f)
        for cone in self._cone_visibility:
            cone.visible = self._compute_cone_is_visible(cone)

        self._s1 = 0.9 * self._s1 + 0.1 * self._s1Command
        self._s2 = 0.9 * self._s2 + 0.1 * self._s2Command

    def _compute_cone_is_visible(self, cone: Cone):

        relativePose = np.linalg.inv(self._pos).dot(cone.pos)

        angle = get2DMatAngle(relativePose)
        midAngle = 30.0 / 180.0 * math.pi
        relativeAngle = abs(angle) / midAngle
        relativeTolerance = 0.1
        if 1 + relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.0
        elif 1 - relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.5
        else:
            angleVisibilityProb = 0.99

        distance = math.sqrt(relativePose[1,2] * relativePose[1,2] + relativePose[0,2] * relativePose[0,2])
        minDistance = 100
        minTolerance = 50
        maxDistance = 600
        maxTolerance = 100
        if distance < minDistance - minTolerance:
            distanceProb = 0.0
        elif distance < minDistance + minTolerance:
            distanceProb = 0.99 * (distance - (minDistance - minTolerance)) / (2 * minTolerance)
        elif distance < maxDistance - maxTolerance:
            distanceProb = 0.99
        elif distance < maxDistance + maxTolerance:
            distanceProb = 0.99 - 0.99 * (distance - (maxDistance - maxTolerance)) / (2 * maxTolerance)
        else:
            distanceProb = 0.0

        p = angleVisibilityProb * distanceProb

        if random.random() < p:
            return True
        return False

    def cone_is_visible(self, cone):
        return self._cone_visibility[cone]

    def drive_wheel_motors(self, l_wheel_speed, r_wheel_speed):
        self._s1Command = max(-100, min(l_wheel_speed, 100))
        self._s2Command = max(-100, min(r_wheel_speed, 100))

    def cone_pose_relative(self, cone: Cone):
        if not self.cone_is_visible(cone):
            return np.identity(3)

        relativePose = np.linalg.inv(self._pos).dot(cone.pos)

        distance = math.sqrt(relativePose[1,2] * relativePose[1,2] + relativePose[0,2] * relativePose[0,2])

        dx = np.random.normal(0, 5) + np.random.normal(0, 5) * (distance / 200)
        dy = np.random.normal(0, 3) + np.random.normal(0, 3) * (distance / 200)
        da = np.random.normal(0, 0.05)

        sensedRelative = transformationMat(relativePose[0,2]+dx, relativePose[1,2]+dy, get2DMatAngle(relativePose)+da)
        return sensedRelative

    def cone_pose_global(self, cone):
        if not self.cone_is_visible(cone):
            return np.identity(3)

        return self._pos.dot(self.cone_pose_relative(cone))

    def left_wheel_speed(self):
        if random.random() < 0.05:
             return 0
        return float(int(self._s1))

    def right_wheel_speed(self):
        if random.random() < 0.05:
             return 0
        return float(int(self._s2))

#===================SENSORS=========================================

    def sensor_left_speed(self):
        if random.random() < 0.05:
             return 0
        return float(int(self._s2))

    def sensor_right_speed(self):
        if random.random() < 0.05:
             return 0
        return float(int(self._s2))

    def sensor_gps(self):
        if random.random() < 0.05:
             return [self._pos[0,2]+random.randint(-10,10), self._pos[1,2]+random.randint(-10,10)]
        return [self._pos[0,2], self._pos[1,2]]

    def sensor_imu(self):
        #TODO
        return 0

    def sensor_camera(self):
        return


def runWorld(w: SimWorld, finished):
    while not finished.is_set():
        t0 = time.time()
        w.sim_step()
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < w.waitTime:
            time.sleep(w.waitTime - timeTaken)




