import numpy as np
import math

#Create a transformation matrix from x,y,angle
def transformationMat(x,y,angle):
    return np.array([[math.cos(angle), -math.sin(angle), x], [math.sin(angle), math.cos(angle), y], [0., 0., 1.]])

#Get angle of rotation in radians
def get2DMatAngle(mat: np.array):
    return math.atan2(mat[1,0], mat[0,0])

#Get rotation matrix from radian angle
def radiansTo2DMatAngle(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
