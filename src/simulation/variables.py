import numpy as np
import math

#==Declare sim program variables==
#sim_world
conf_cameraAngle = 180
conf_cameraMinD = 0.5
conf_cameraMinTol = 0.25
conf_cameraMaxD = 5
conf_cameraMaxTol = 0.5

#Track


#driving_commands
conf_wheelDistance = 0.08

#runSimTrack

#===============Common_Functions================================

#Create a transformation matrix from x,y,angle
def transformationMat(x,y,angle):
    return np.array([[math.cos(angle), -math.sin(angle), x], [math.sin(angle), math.cos(angle), y], [0., 0., 1.]])

#Get angle of rotation in radians
def get2DMatAngle(mat: np.array):
    return math.atan2(mat[1,0], mat[0,0])

#Get rotation matrix from radian angle
def radiansTo2DMatAngle(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
