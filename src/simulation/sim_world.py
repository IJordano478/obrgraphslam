from src.simulation.frame2d import Frame2D

from src.simulation.Track import Track, Coord2D

import math
import numpy as np
import random
import time


# Take a true cone position (relative to robot frame). 
# Compute /probability/ of cone being (i) visible AND being detected at a specific measure position (relative to robot frame)
#def cone_sensor_model(trueconePosition, visible, measuredPosition):
#    sigma = np.array([20.0, 20.0, 10.0 / 180 * math.pi])
#    sigSquareInv = 1.0 / (sigma * sigma)
#    if visible:
#        # x/y/a Gaussian
#        deviation = measuredPosition.inverse().mult(trueconePosition)
#        xya = deviation.toXYA()
#        error = 0.5 * np.sum(xya * sigSquareInv * xya)
#        positionProb = math.exp(-error)
#        return positionProb
#    else:
#        # no judgement based purely on visibility at this point
#        return 1


#def cozmo_sensor_model(robotPose: Frame2D, m: CozmoMap, cliffDetected, coneVisibility, coneRelativeFrames):
#    p = 1.
#    for coneID in coneVisibility:
#        relativePose = coneRelativeFrames[coneID]
#        relativeTruePose = robotPose.inverse().mult(m.landmarks[coneID])
#        p = p * cone_sensor_model(relativeTruePose, coneVisibility[coneID], relativePose)
#    p = p * cozmo_cliff_sensor_model(robotPose, m, cliffDetected)
#    return p


class SimWorld:
    def __init__(self, m, pos):
        self.map = m
        self._pos = pos
        self._s = 0
        self._sCommand = 0
        self.waitTime = 0.05
        self._cone_visibility = self._init_landmarks(m.landmarks)

    def _init_landmarks(self, landmarks):
        for row in landmarks:
            row.append(False)
        return landmarks

    def _dont_touch__pos(self):
        return self._pos

    def _dont_touch__step(self):
        x = self._pos.x()
        y = self._pos.y()

        s = self._s * self.waitTime
        a = s / (np.exp(4.44265125649))
        if abs(a) > 0.0001:
            rad = s / (2 * a)
            x = rad * math.sin(a)
            y = -rad * (math.cos(a) - 1)
        else:
            x = s
            y = 0
        d = self._s - self._sCommand
        accel = d * d
        # multiplicative noise on x and a, scaled by acceleration, to simulate slippage
        x = x + x * np.random.normal(0, min(0.5, 0.0001 * accel))
        a = a + a * np.random.normal(0, min(0.5, 0.0001 * accel))
        f = Frame2D.fromXYA(x, y, a)

        self._pos = self._pos.mult(f)

        for cone in self._cone_visibility:
            cone[3] = self._compute_cone_is_visible(cone)
        #TODO change for cone sensor

        self._s = 0.9 * self._s + 0.1 * self._sCommand

    #TODO _compute_cone_is_visible
    def _compute_cone_is_visible(self, cone):
        relativePose = self._pos.inverse().mult(self.map.landmarks[cone])
        angle = math.atan2(relativePose.y(), relativePose.x())
        midAngle = 30.0 / 180.0 * math.pi
        relativeAngle = abs(angle) / midAngle
        relativeTolerance = 0.1
        if 1 + relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.0
        elif 1 - relativeTolerance < relativeAngle:
            angleVisibilityProb = 0.5
        else:
            angleVisibilityProb = 0.99

        distance = math.sqrt(relativePose.y() * relativePose.y() + relativePose.x() * relativePose.x())
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

    def cone_is_visible(self, coneID):
        return self._cone_visibility[coneID]

    def drive_wheel_motors(self, wheel_speed):
        self._sCommand = max(-100, min(wheel_speed, 100))

    def cone_pose_relative(self, coneID):
        if not self.cone_is_visible(coneID):
            return Frame2D()

        relativePose = self._pos.inverse().mult(self.map.landmarks[coneID])

        distance = math.sqrt(relativePose.y() * relativePose.y() + relativePose.x() * relativePose.x())

        dx = np.random.normal(0, 5) + np.random.normal(0, 5) * (distance / 200)
        dy = np.random.normal(0, 3) + np.random.normal(0, 3) * (distance / 200)
        da = np.random.normal(0, 0.05)

        sensedRelative = Frame2D.fromXYA(relativePose.x() + dx, relativePose.y() + dy, relativePose.angle() + da)

        return sensedRelative

    def cone_pose_global(self, coneID):
        if not self.cone_is_visible(coneID):
            return Frame2D()

        return self._pos.mult(self.cone_pose_relative(coneID))

    def wheel_speed(self):
        if random.random() < 0.05:
             return 0
        return float(int(self._s))

def runWorld(w: SimWorld, finished):
    while not finished.is_set():
        t0 = time.time()
        w.dont_touch__step()
        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < w.waitTime:
            time.sleep(w.waitTime - timeTaken)




