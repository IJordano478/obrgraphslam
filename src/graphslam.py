#!/usr/bin/env python

import rospy
from 
import time

rospy.init_node('graphSLAM', anonymous=True)
velocityPub = rospy.Publisher('/obr_msgs/velocity', )

class GraphSLAM:
    def __init__(self):
        # initialise variables as self.variable
        # TODO

    def reset(self):
        # TODO

    def listenCones(self, coneArray):
        # TODO

    def listenGPS(self, gpsTruth):
        # TODO

    def listenIMU(self, imuTruth):
        # TODO

    def listenIMU(self, imuTruth):
        # TODO

    def publishDrivingCommand(self):
        # TODO

    def main(self):
        # TODO: controller


if __name__ == '__main__':
    rospy.init_node('graphslam', anonymous=False)
    graphslam = GraphSLAM()
    rospy.Subscriber("/obr_msgs/msg/ConeArray", ConeArray, graphslam.listenCones())
    rospy.Subscriber("/peak_gps/gps/truth", NavSatFix, graphslam.listenGPS())
    rospy.Subscriber("/peak_gps/imu/truth", Imu, graphslam.listenIMU())
    rospy.loginfo("Created subscribers: ConeArray, gps-truth, imu-truth")

    rospy.spin()
