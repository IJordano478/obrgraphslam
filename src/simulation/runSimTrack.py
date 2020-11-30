from src.simulation.Track import Track, plotTrack, loadTrack
from src.simulation.variables import *
from src.simulation.driving_commands import *
from src.simulation.sim_world import *
from matplotlib import pyplot as plt
#from terminal_reader import

import asyncio
import math
import numpy as np
import threading
import time

# this data structure represents the map
m = loadTrack("Oval.csv")
currentPose = transformationMat(0,0,0)
TargetPose = transformationMat(0,0,0)

#Plots the OBR car in matplot
def plotRobot(pos: np.array, colour="orange", existingPlot=None):
    scale = 0.005
    xy = np.array([[150, 150, 90, 60, 10, -40, -70, -90, -130, -130, -90, -70, -40,  10,  60,  90, 150],
                   [-20, 20,  40, 40, 25,  25,  40,  40,   20,  -20, -40, -40, -25, -25, -40, -40, -20]])
    ones = np.ones((1,17))

    xy = xy*scale
    xy = np.vstack((xy, ones))
    xy = np.matmul(pos, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(colour)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], colour)
        return line[0]

#Plots all the cones in matplot
def plotLandmark(cone: Cone, color="orange", existingPlot=None):
    xy = np.array([[25, -25, -25, 25, 25],
                   [25, 25, -25, -25, 25],
                   [1, 1, 1, 1, 1]])

    if cone.pos[0,2] == 0.0 and cone.pos[1,2] == 0.0:
        xy = np.matmul(np.array([[math.cos(0), -math.sin(0), -1000], [math.sin(0), math.cos(0), -1000], [0., 0.,1.]]), xy)
    else:
        xy = np.matmul(cone.pos, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(cone.colour)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], cone.colour)
        return line[0]

#Keeps updating the same matplot until the program ends
def runPlotLoop(simWorld: SimWorld, finished):
    global particles

    # create plot
    plt.ion()
    plt.figure()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.set_xlim(m.minX, m.maxX)
    ax.set_ylim(m.minY, m.maxY)

    plotTrack(ax, m, True)

    robotPlot = plotRobot(simWorld.sim_get_pos())
    plt.pause(0.01)
    landmarkPlots = []
    for i in range(0, len(simWorld._cone_visibility)):
        landmarkPlots.append(plotLandmark(simWorld._cone_visibility[i]))
    plt.pause(0.01)
    # main loop
    t = 0
    while not finished.is_set():
        # update plot
        plotRobot(simWorld.sim_get_pos(), existingPlot=robotPlot)
        plt.pause(0.01)
        for i in range(0, len(simWorld._cone_visibility)):
            plotLandmark(simWorld._cone_visibility[i], existingPlot=landmarkPlots[i])

        plt.draw()
        plt.pause(0.01)
        time.sleep(0.01)


#Sets the car to keep driving a specified path. When SLAM is done you can modify this ;)
def runDriveLoop(simWorld: SimWorld, finished):
    global currentPose
    global currentTarget

    #The points that the car will follow. If you load in your own track be sure to change these to match it. Also be
    #mindful of rotation (in radians). **ALSO** as using a spline technique for driving, weird things may happen in
    #some situations so if occurs just move the points very slightly (singularities occur annoyingly often)
    pathNodes = {
        "A": transformationMat(8, 14, math.pi/2),
        "B": transformationMat(5, 16, math.pi),
        "C": transformationMat(2, 14, math.pi*3/2),
        "D": transformationMat(2, 9, math.pi*3/2),
        "E": transformationMat(2, 4,  math.pi*3/2),
        "F": transformationMat(5, 2,  math.pi*2),
        "G": transformationMat(8, 4,  math.pi/2),
        "H": transformationMat(8, 9, math.pi/2)}
    currentTarget = pathNodes["A"]
    time.sleep(5)

    # main loop

    while (True):
        currentPose = simWorld.sim_get_pos()
        X = currentPose[0, 2]
        Y = currentPose[1, 2]

        keys = list(pathNodes.keys())
        for key_index, key in enumerate(keys):
            if (X - pathNodes[key][0, 2]) ** 2 + (Y - pathNodes[key][1, 2]) ** 2 < 0.25:
                currentTarget = pathNodes[keys[(key_index + 1) % len(keys)]]
                #print("Changing target")
                break

        #print("GPS:", simWorld.sensor_gps())
        #print("Speed:", simWorld.sensor_left_speed(), ",", simWorld.sensor_right_speed())
        #print("Cones:", simWorld.sensor_camera())

        # Set route
        relativeTarget = np.matmul(np.linalg.inv(currentPose), currentTarget)
        velocity = target_pose_to_velocity_spline(relativeTarget)
        trackSpeed = velocity_to_track_speed(velocity[0], velocity[1])
        simWorld.drive_wheel_motors(trackSpeed[0], trackSpeed[1])
        time.sleep(0.2)

        # Set currentPose
        delta = track_speed_to_pose_change(trackSpeed[0], trackSpeed[1], 0.2)
        currentPose = np.matmul(currentPose,delta)
    # Cozmo is at target
    simWorld.drive_wheel_motors(0, 0)
    print("finished")

#Create the threads to run everything
def cozmo_program(simWorld: SimWorld):
    finished = threading.Event()
    print("Starting simulation. Press Q to exit", end="\r\n")
    threading.Thread(target=runWorld, args=(simWorld, finished)).start()
    #threading.Thread(target=WaitForChar, args=(finished, '[Qq]')).start() #TODO ...perhaps
    threading.Thread(target=runDriveLoop, args=(simWorld, finished)).start()
    # running the plot loop in a thread is not thread-safe because matplotlib
    # uses tkinter, which in turn has a threading quirk that makes it
    # non-thread-safe outside the python main program.
    # See https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop

    #threading.Thread(target=runPlotLoop, args=(simWorld,finished)).start()
    runPlotLoop(simWorld, finished)

#Opening to simulation, here you can specify a start X and Y
startX = 8.0
startY = 5.0
startA = math.pi/2
startPos = transformationMat(startX, startY, startA)
currentPose = startPos

simWorld = SimWorld(m, startPos)
cozmo_program(simWorld)
