# Copyright (c) 2019 Matthias Rolf, Oxford Brookes University

'''

'''

#from cozmo.util import Pose, Position

import numpy as np
import math


class Frame2D:
    mat = np.matrix([])

    # mat = np.matrix([[1., 0.,0.], [0., 1.,0.], [0., 0.,1.]])   #added in to solve errors
    def __init__(self):
        self.mat = np.matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    @classmethod
    def fromMat(cls, m: np.matrix):
        f = cls()
        f.mat = m
        return f

    #@classmethod
    #def fromPose(cls, pose: Pose):
    #    f = cls()
    #    angle = pose.rotation.angle_z.radians
    #    f.mat = np.matrix(
    #        [[math.cos(angle), -math.sin(angle), pose.position.x], [math.sin(angle), math.cos(angle), pose.position.y],
    #         [0., 0., 1.]])
    #    return f;

    # create a frame from x/y offset and angle a
    @classmethod
    def fromXYA(cls, x, y=None, a=None):
        if y is not None:
            f = cls()
            f.mat = np.matrix([[math.cos(a), -math.sin(a), x], [math.sin(a), math.cos(a), y], [0., 0., 1.]])
            return f
        else:
            f = cls()
            xm = x.reshape(-1)
            f.mat = np.matrix(
                [[math.cos(xm[2]), -math.sin(xm[2]), xm[0]], [math.sin(xm[2]), math.cos(xm[2]), xm[1]], [0., 0., 1.]])
            return f

    def toXYA(self):
        return np.array([self.mat[0, 2], self.mat[1, 2], self.angle()])

    def __str__(self):
        return "[x=" + str(self.mat[0, 2]) + ",y=" + str(self.mat[1, 2]) + ",a=" + str(self.angle()) + "]"

    # @classmethod
    def inverse(self):
        return Frame2D.fromMat(np.linalg.inv(self.mat))

    # Coordinate frame concatenation by means of matrix product
    def mult(self, other):
        f = Frame2D()
        f.mat = np.matmul(self.mat, other.mat)
        return f;

    def x(self):
        return self.mat[0, 2]

    def y(self):
        return self.mat[1, 2]

    def angle(self):
        return math.atan2(self.mat[1, 0], self.mat[0, 0])

