import asyncio

from src.simulation.Track import Track, plotTrack, loadTrack
from matplotlib import pyplot as plt
#from cozmo_interface import *
from src.simulation.sim_world import *
#from terminal_reader import WaitForChar
import math
import numpy as np
import threading
import time

# this data structure represents the map
m = loadTrack("Oval.csv")

# noise injected in re-sampling process to avoid multiple exact duplications of a particle
#xyaResampleVar = np.diag([300, 300, 0.5 * math.pi / 180])
# note here: instead of creating new gaussian random numbers every time, which is /very/ expensive,
# 	precompute a large table of them an recycle. GaussianTable does that internally
#xyaResampleNoise = GaussianTable(np.zeros([3]), xyaResampleVar, 10000)

# Motor error model
#cozmoOdomNoiseX = 0.2
#cozmoOdomNoiseY = 0.2
#cozmoOdomNoiseTheta = 0.001
#xyaNoiseVar = np.diag([cozmoOdomNoiseX, cozmoOdomNoiseY, cozmoOdomNoiseTheta])
#xyaNoise = GaussianTable(np.zeros([3]), xyaNoiseVar, 10000)

currentPose = np.identity(3)


def plotRobot(pos: np.array, colour="orange", existingPlot=None):
    #xy = np.array([[3, 3.5, 3.5, 3, -3, -3, 3],
    #               [2, 1.5, -1.5, -2, -2, 2, 2],
    #               [1, 1, 1, 1, 1, 1, 1]])
    scale = 0.005
    xy = np.array([[20, -20, -40, -40, -25, -25, -40, -40, -20,    20,  40,  40,  25, 25, 40, 40, 20],
                   [150,  150,  90,  60,  10, -40, -70, -90, -130, -130, -90, -70, -40, 10, 60, 90, 150]])
    ones = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    xy = xy*scale
    xy = np.vstack((xy, ones))
    #print(xy)
    #breakpoint()
    xy = np.matmul(pos, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(colour)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], colour)
        return line[0]


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

    robotPlot = plotRobot(simWorld._dont_touch__pos())
    plt.pause(0.01)
    landmarkPlots = []
    for i in range(0, len(simWorld._cone_visibility)):
        landmarkPlots.append(plotLandmark(simWorld._cone_visibility[i]))
    plt.pause(0.01)
    # main loop
    t = 0
    while not finished.is_set():
        # update plot
        plotRobot(simWorld._dont_touch__pos(), existingPlot=robotPlot)
        plt.pause(0.01)
        for i in range(0, len(simWorld._cone_visibility)):
            plotLandmark(simWorld._cone_visibility[i], existingPlot=landmarkPlots[i])

        plt.draw()
        plt.pause(0.01)

        time.sleep(0.01)


def runMainLoop(simWorld: SimWorld, finished):
    print("finished")


def cozmo_program(simWorld: SimWorld):
    finished = threading.Event()
    print("Starting simulation. Press Q to exit", end="\r\n")
    threading.Thread(target=runWorld, args=(simWorld, finished)).start()
    #threading.Thread(target=WaitForChar, args=(finished, '[Qq]')).start()
    #threading.Thread(target=runMainLoop, args=(simWorld, finished)).start()
    # running the plot loop in a thread is not thread-safe because matplotlib
    # uses tkinter, which in turn has a threading quirk that makes it
    # non-thread-safe outside the python main program.
    # See https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop

    # threading.Thread(target=runPlotLoop, args=(simWorld,finished)).start()
    runPlotLoop(simWorld, finished)


# NOTE: this code allows to specify the initial position of Cozmo on the map
startX = 8.0
startY = 5.0
startA = 0
startPos = np.array([[math.cos(startA), -math.sin(startA), startX], [math.sin(startA), math.cos(startA), startY], [0., 0.,1.]])
simWorld = SimWorld(m, startPos)

cozmo_program(simWorld)
