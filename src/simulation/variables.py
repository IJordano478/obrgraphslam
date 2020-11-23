import numpy as np
import math

def transformationMat(x,y,angle):
    return np.array([[math.cos(angle), -math.sin(angle), x], [math.sin(angle), math.cos(angle), y], [0., 0., 1.]])

def get2DMatAngle(mat: np.array):
    return math.atan2(mat[1,0], mat[0,0])

def radiansTo2DMatAngle(angle):
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
