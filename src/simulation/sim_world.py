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

#Simulates a single step in the simulation; so updates all the information to the new values
    def sim_step(self):
        x = self._pos[0,2]
        y = self._pos[1,2]

        s1 = self._s1 * self.waitTime
        s2 = self._s2 * self.waitTime
        #a = (s2-s1) / (np.exp(4.44265125649))
        a = (s2 - s1) / conf_wheelDistance
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

        f = transformationMat(x, y, a)
        #f = np.array([[math.cos(a), -math.sin(a), x], [math.sin(a), math.cos(a), y], [0., 0.,1.]])
        self._pos = np.matmul(self._pos, f)
        for cone in self._cone_visibility:
            cone.visible = self._compute_cone_is_visible(cone)

        self._s1 = 0.9 * self._s1 + 0.1 * self._s1Command
        self._s2 = 0.9 * self._s2 + 0.1 * self._s2Command

#Checks each cone based on the angle and distance between them to determine visibility
    def _compute_cone_is_visible(self, cone: Cone):

        relativePose = np.matmul(np.linalg.inv(self._pos), cone.pos)

        angle = math.atan2(relativePose[1,2], relativePose[0,2])

        # camera field of view (degrees)
        midAngle = conf_cameraAngle / 180.0 * math.pi
        relativeAngle = abs(angle) / midAngle

        relativeTolerance = 0.1
        if 1 + relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.0
        elif 1 - relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.5
        else:
            angleVisibilityProb = 0.99

        distance = math.sqrt(relativePose[1,2] * relativePose[1,2] + relativePose[0,2] * relativePose[0,2])
        minDistance = conf_cameraMinD
        minTolerance = conf_cameraMinTol
        maxDistance = conf_cameraMaxD
        maxTolerance = conf_cameraMaxTol
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
        return float(self._s1)

    def right_wheel_speed(self):
        if random.random() < 0.05:
             return 0
        return float(self._s2)

#===================SENSORS=============================================
#For anyone working on SLAM use these to simulate listening to a ROStopic
#Note: sensors include some simulated noise (based on random factor), you can turn it off or work with it.
#Having noise is preferable however it can be more experimental to turn it off if you are specifically testing
#an area

    def sensor_left_speed(self):
        if random.random() < 0.05:
             return 0
        return round(float(self._s1),3)

    def sensor_right_speed(self):
        if random.random() < 0.05:
             return 0
        return round(float(self._s2),3)

    def sensor_gps(self):
        if random.random() < 0.05:
             return [self._pos[0,2]+random.randint(-10,10), self._pos[1,2]+random.randint(-10,10)]
        return [self._pos[0,2], self._pos[1,2]]

    def sensor_imu(self):
        #TODO I'll sort this out when I know what our IMU will do
        return 0

    def sensor_camera(self):
        detected = np.array([])
        for cone in self._cone_visibility:
            if(cone.visible==True):
                if detected.size == 0:
                    detected = np.array([[cone.pos[0,2]],[cone.pos[1,2]]])
                    #print(detected)
                else:
                    new = np.array([[cone.pos[0, 2]], [cone.pos[1, 2]]])
                    detected = np.concatenate((detected,new), axis=1)
        return detected


def runWorld(w: SimWorld, finished):
    while not finished.is_set():
        t0 = time.time()
        w.sim_step()
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < w.waitTime:
            time.sleep(w.waitTime - timeTaken)




