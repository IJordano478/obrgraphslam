#GraphSLAM

GraphSLAM implementation for a Lidar based autonomous driving robot

Based on theory and algorithms presented in Probabilistic Robotics (2005), Thrun, S. , Burgard, W. , and Fox, D. 

Contains 2 parts, the offline graphSLAM (semi-discontinued) and online SEIF (current focus) algorithms

Current status:
 - Working methods for a known correspondence SEIF
 - Allows repeated calls of methods
 - Unknown correspondence version 1 designed but not included
 - python simulator allows real-time simulation of algorithm
 - SEIF for ROS2 included
 
 

TODO:
 - Improve efficiency of code, some functions are still inverting large matrices, causing code to run slowly with large numbers of landmarks
 - Correct GPS sensor fusion error in measurements
 - Design better result plotter, currently componentTester.py generates graphs from a .csv file, results should be displayed after program finishes
