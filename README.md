#GraphSLAM

GraphSLAM implementation for a Lidar based autonomous driving robot

Based on theory and algorithms presented in Probabilistic Robotics (2005), Thrun, S. , Burgard, W. , and Fox, D. 

Contains 2 parts, the offline graphSLAM (semi-discontinued) and online SEIF (current focus) algorithms

Current status:
 - Working methods for a known correspondence SEIF
 - Allows repeated calls of methods
 

TODO:
 - Adapt code to include signiture of landmark in decision making
 - Design unknown correspondence method
 - Adapt code for a single loop for the SEIF algorithm, allowing passing of data through global variables
 - Tie in to simulation
 - Improve efficiency of code, python matrix manipulation renders the use of projection matrices redundant and will 
 improve calculation speeds