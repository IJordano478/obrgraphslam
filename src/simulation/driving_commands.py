#!/usr/bin/env python3

from src.simulation.variables import *

import math
import numpy as np


wheelDistance = conf_wheelDistance

cozmoOdomNoiseX = 0.01
cozmoOdomNoiseY = 0.01
cozmoOdomNoiseTheta = 0.01


# Forward kinematics: compute coordinate frame update as transformation matrix from left/right track speed and
# time of movement
def track_speed_to_pose_change(left, right, time):
    # left and right as speeds, time as interval for loops"

    # track speed * time = distance travelled
    dLeft = left * time
    dRight = right * time

    theta = (dRight - dLeft) / wheelDistance
    if -0.05 < theta < 0.05:
        r = 0
        d = np.array([[dLeft], [0]])
    else:
        r = (dLeft + dRight) / (2 * theta)
        d = np.array([[r * math.sin(theta)], [(-r) * (math.cos(theta) - 1)]])

    mat = transformationMat(d[0, 0], d[1, 0], theta)
    return mat


# Differential inverse kinematics: compute left/right track speed from desired angular and forward velocity
def velocity_to_track_speed(forward, angular):
    vLeft = forward - (angular * (wheelDistance / 2))
    vRight = forward + (angular * (wheelDistance / 2))
    return [vLeft, vRight]


# Trajectory planning: given target (relative to robot frame), determine next forward/angular motion
# Implement by means of cubic spline interpolation
def target_pose_to_velocity_spline(relativeTarget: np.array):
    velocity = 0
    angular = 0
    dx = relativeTarget[0, 2]
    dy = relativeTarget[1, 2]
    da = math.atan2(dy, dx)
    dOrient = get2DMatAngle(relativeTarget)
    dError = 0
    aError = 0.1

    # speed = dx   #CHOOSE   "Parametrized speed or slope s at beginning ... set to distance of target"  used 100 now using dx
    speed = math.sqrt(dx * dx + dy * dy) # using this value as if the target is directly left or right of cozmo k = inf

    k = (2 * (3 * dy - speed * relativeTarget[1, 0])) / (speed * speed)

    # l = w*r = w*(1/k)
    l = 1  # CHOOSE   Start by choosing a single value that will be how quickly i want to go, small makes it creep, big goes fast

    # Stop cozmo when the curve changes too quickly
    # prevent cozmo doing more than pi/8 radians in 0.1 seconds
    #if (k >= 0.4 or k <= -0.4):
    if (k >= 1 or k <= -1):

        if (k >= 1):
            angular = 0.1
        else:
            angular = -0.1
        velocity = 0
    else:

        # angular = l*k
        # velocity = l
        angular = l * k
        velocity = l
    return [velocity, angular]


# Trajectory planning: given target (relative to robot frame), determine next forward/angular motion
# Implement in a linear way
# If far away and facing wrong direction: rotate to face target
# If far away and facing target: move forward
# If on target: turn to desired orientation
def target_pose_to_velocity_linear(relativeTarget: np.array):
    velocity = 0
    angular = 0
    dx = relativeTarget[0, 2]
    dy = relativeTarget[1, 2]
    da = math.atan2(dy, dx)
    dOrient = get2DMatAngle(relativeTarget)
    dError = 0
    aError = 0.05

    # if(dx*dx + dy*dy < dError*dError):
    if ((dx > -dError and dx < dError) and (dy > -dError and dy < dError) and (
            dOrient > -aError and dOrient < aError)):
        velocity = 0
        angular = 0
        # print("Finished")
    # if on target x and y
    elif ((dx > -dError and dx < dError) and (dy > -dError and dy < dError)):
        velocity = 0
        angular = dOrient / 100
        # print("Re-orienting")
    # if not on target and direction lines up
    elif (((dx < -dError or dx > dError) or (dy < -dError or dy > dError)) and (da > -aError and da < aError)):
        velocity = dx/2
        angular = 0
        # print("Driving to target")
    # else not on target and direction not aligned
    else:
        velocity = 0
        angular = da/100
        # print("Turning to target")

    return [velocity, angular]
