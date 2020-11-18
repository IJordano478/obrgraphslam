import asyncio

from frame2d import Frame2D
from src.simulation.Track import Track, plotTrack, loadTrack, Coord2D
from matplotlib import pyplot as plt
from cozmo_interface import *
from mcl_tools import *
from cozmo_sim_world import *
from terminal_reader import WaitForChar
from gaussian import Gaussian, GaussianTable, plotGaussian
import math
import numpy as np
import threading
import time

# this data structure represents the map
m = loadTrack("Oval.csv")

# this probability distribution represents a uniform distribution over the entire map in any orientation
mapPrior = Uniform(
    np.array([m.grid.minX(), m.grid.minY(), 0]),
    np.array([m.grid.maxX(), m.grid.maxY(), 2 * math.pi]))

numParticles = 30

# The main data structure: array for particles, each represnted as Frame2D     //particles = [Frame2D]
particles = sampleFromPrior(mapPrior, numParticles)

# noise injected in re-sampling process to avoid multiple exact duplications of a particle
xyaResampleVar = np.diag([300, 300, 0.5 * math.pi / 180])
# note here: instead of creating new gaussian random numbers every time, which is /very/ expensive,
# 	precompute a large table of them an recycle. GaussianTable does that internally
xyaResampleNoise = GaussianTable(np.zeros([3]), xyaResampleVar, 10000)

# Motor error model
cozmoOdomNoiseX = 0.2
cozmoOdomNoiseY = 0.2
cozmoOdomNoiseTheta = 0.001
xyaNoiseVar = np.diag([cozmoOdomNoiseX, cozmoOdomNoiseY, cozmoOdomNoiseTheta])
xyaNoise = GaussianTable(np.zeros([3]), xyaNoiseVar, 10000)

currentPose = Frame2D()
currentTarget = Frame2D.fromXYA(0, 0, 0)


def runMCLLoop(simWorld: CozmoSimWorld, finished):
    global particles

    particleWeights = np.zeros([numParticles])
    cubeIDs = [cozmo.objects.LightCube1Id, cozmo.objects.LightCube2Id, cozmo.objects.LightCube3Id]

    # main loop
    timeInterval = 0.2
    t = 0
    while not finished.is_set():
        t0 = time.time()

        # read cube sensors
        cubeVisibility = {}
        cubeRelativeFrames = {}
        numVisibleCubes = 0
        for cubeID in cubeIDs:
            relativePose = Frame2D()
            visible = False
            if simWorld.cube_is_visible(cubeID):
                relativePose = simWorld.cube_pose_relative(cubeID)
                visible = True
                numVisibleCubes = numVisibleCubes + 1
            cubeVisibility[cubeID] = visible
            cubeRelativeFrames[cubeID] = relativePose

        # read cliff sensor
        cliffDetected = simWorld.is_cliff_detected()

        # read track speeds
        lspeed = simWorld.left_wheel_speed()
        rspeed = simWorld.right_wheel_speed()
        poseChange = track_speed_to_pose_change(lspeed, rspeed, timeInterval)
        poseChangeXYA = poseChange.toXYA()

        # read global variable
        currentParticles = particles

        # MCL step 1: prediction (shift particle through motion model)
        for i in range(0, numParticles):
            poseChangeXYAnoise = np.add(poseChangeXYA, xyaNoise.sample())
            poseChangeNoise = Frame2D.fromXYA(poseChangeXYAnoise)
            currentParticles[i] = currentParticles[i].mult(poseChangeNoise)

        # MCL step 2: weighting (weigh particles with sensor model)
        for i in range(0, numParticles):
            particleWeights[i] = cozmo_cliff_sensor_model(currentParticles[i], m, cliffDetected)
            particleWeights[i] = cozmo_sensor_model(
                currentParticles[i],
                m,
                cliffDetected,
                cubeVisibility,
                cubeRelativeFrames
            )

        # MCL step 3: resampling (proportional to weights)
        numFreshSamples = 30
        currentParticles.append(sampleFromPrior(mapPrior, numFreshSamples))
        newParticles = resampleLowVar(currentParticles, particleWeights, numParticles, xyaResampleNoise)

        # write global variable
        particles = newParticles

        print("t = " + str(t), end="\r\n")
        t = t + 1

        t1 = time.time()
        timeTaken = t1 - t0
        if timeTaken < timeInterval:
            time.sleep(timeInterval - timeTaken)
        else:
            print("Warning: loop iteration tool more than " + str(timeInterval) + " seconds (t=" + str(timeTaken) + ")",
                  end="\r\n")


def plotRobot(pos: Frame2D, color="orange", existingPlot=None):
    xy = np.array([[30, 35, 35, 30, -30, -30, 30],
                   [20, 15, -15, -20, -20, 20, 20],
                   [1, 1, 1, 1, 1, 1, 1]])
    xy = np.matmul(pos.mat, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(color)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], color)
        return line[0]


def plotLandmark(pos: Frame2D, color="orange", existingPlot=None):
    xy = np.array([[25, -25, -25, 25, 25],
                   [25, 25, -25, -25, 25],
                   [1, 1, 1, 1, 1]])
    if pos.x() == 0 and pos.y() == 0:
        xy = np.matmul(Frame2D.fromXYA(-1000, -1000, 0).mat, xy)
    else:
        xy = np.matmul(pos.mat, xy)
    if existingPlot is not None:
        existingPlot.set_xdata(xy[0, :])
        existingPlot.set_ydata(xy[1, :])
        existingPlot.set_color(color)
        return existingPlot
    else:
        line = plt.plot(xy[0, :], xy[1, :], color)
        return line[0]


def runPlotLoop(simWorld: CozmoSimWorld, finished):
    global particles

    # create plot
    plt.ion()
    plt.show()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, aspect=1)

    ax.set_xlim(m.grid.minX(), m.grid.maxX())
    ax.set_ylim(m.grid.minY(), m.grid.maxY())

    plotMap(ax, m)

    particlesXYA = np.zeros([numParticles, 3])
    for i in range(0, numParticles):
        particlesXYA[i, :] = particles[i].toXYA()
    particlePlot = plt.scatter(particlesXYA[:, 0], particlesXYA[:, 1], color="red", zorder=3, s=10, alpha=0.5)

    empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
    print(str(empiricalG))
    gaussianPlot = plotGaussian(empiricalG, color="red")

    robotPlot = plotRobot(simWorld.dont_touch__pos())
    cube1Plot = plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube1Id))
    cube2Plot = plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube2Id))
    cube3Plot = plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube3Id))

    # main loop
    t = 0
    while not finished.is_set():
        # update plot
        for i in range(0, numParticles):
            particlesXYA[i, :] = particles[i].toXYA()
        particlePlot.set_offsets(particlesXYA[:, 0:2])

        empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
        plotGaussian(empiricalG, color="red", existingPlot=gaussianPlot)

        plotRobot(simWorld.dont_touch__pos(), existingPlot=robotPlot)
        plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube1Id), existingPlot=cube1Plot)
        plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube2Id), existingPlot=cube2Plot)
        plotLandmark(simWorld.cube_pose_global(cozmo.objects.LightCube3Id), existingPlot=cube3Plot)

        plt.draw()
        plt.pause(0.001)

        time.sleep(0.01)


def runCozmoMainLoop(simWorld: CozmoSimWorld, finished):
    global currentPose
    global currentTarget

    pathNodes = {
        "A": Frame2D.fromXYA(300, 200, math.pi / 2),
        "B": Frame2D.fromXYA(200, 400, math.pi / 2),
        "C": Frame2D.fromXYA(300, 600, math.pi / 4),
        "D": Frame2D.fromXYA(400, 760, math.pi / 2)}

    time.sleep(5)

    # main loop

    # Beginning initial data collection
    simWorld.drive_wheel_motors(-30, 30)
    time.sleep(10)
    simWorld.drive_wheel_motors(0, 0)
    time.sleep(1)

    particlesXYA = np.zeros([numParticles, 3])
    totalA = 0
    for i in range(0, numParticles):
        particlesXYA[i, :] = particles[i].toXYA()
        totalA += particles[i].angle()
    empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
    meanX = empiricalG.mean.reshape(-1)[0]
    meanY = empiricalG.mean.reshape(-1)[1]
    meanA = totalA / numParticles
    var = empiricalG.var[0:2, 0:2]

    currentPose = Frame2D.fromXYA(meanX, meanY, meanA)
    print("CurrentPose: (" + str(meanX) + "," + str(meanY) + ") w/ angle: (" + str(meanA) + ")\r\n")

    relativeTarget = Frame2D()
    relativeTarget = currentPose.inverse().mult(pathNodes["D"])

    while ((relativeTarget.x() * relativeTarget.x()) + (relativeTarget.y() * relativeTarget.y()) > 25 * 25):

        if ((meanY < 200 - 25) or ((meanY < 400) and (meanX > 300 + 25))):
            if (currentTarget != pathNodes["A"]):
                print("New target set: Node A")
                currentTarget = pathNodes["A"]
        elif (meanY < 400 - 25):
            if (currentTarget != pathNodes["B"]):
                print("New target set: Node B")
                currentTarget = pathNodes["B"]
        elif (meanY < 600 - 25):
            if (currentTarget != pathNodes["C"]):
                print("New target set: Node C")
                currentTarget = pathNodes["C"]
        else:
            if (currentTarget != pathNodes["D"]):
                print("New target set: Node D")
                currentTarget = pathNodes["D"]

        # Check for edge
        if (simWorld.is_cliff_detected()):
            simWorld.drive_wheel_motors(-10, -10)
            time.sleep(5)
            simWorld.drive_wheel_motors(-20, 20)
            time.sleep(10)

        # if guassian has wide distribution, relocate
        elif ((abs(var[0][0] - var[0][1]) > 5000) or (abs(var[1][0] - var[1][1]) > 10000)):
            simWorld.drive_wheel_motors(-30, 30)
            time.sleep(10)
            simWorld.drive_wheel_motors(0, 0)
            time.sleep(1)

        # If everything is OK, set route
        else:
            relativeTarget = currentPose.inverse().mult(currentTarget)
            # print("relativeTarget"+str(relativeTarget)+"\r")

            velocity = target_pose_to_velocity_linear(relativeTarget)
            # print("velocity"+str(velocity))
            trackSpeed = velocity_to_track_speed(velocity[0], velocity[1])
            simWorld.drive_wheel_motors(trackSpeed[0], trackSpeed[1])
            time.sleep(0.2)

        particlesXYA = np.zeros([numParticles, 3])
        totalA = 0
        for i in range(0, numParticles):
            particlesXYA[i, :] = particles[i].toXYA()
            totalA += particles[i].angle()
        empiricalG = Gaussian.fromData(particlesXYA[:, 0:2])
        meanX = empiricalG.mean.reshape(-1)[0]
        meanY = empiricalG.mean.reshape(-1)[1]
        meanA = totalA / numParticles
        var = empiricalG.var[0:2, 0:2]

        # Set currentPose
        currentPose = Frame2D.fromXYA(meanX, meanY, meanA)

    # Cozmo is at target
    simWorld.drive_wheel_motors(0, 0)
    print("finished")


def cozmo_program(simWorld: CozmoSimWorld):
    finished = threading.Event()
    print("Starting simulation. Press Q to exit", end="\r\n")
    threading.Thread(target=runWorld, args=(simWorld, finished)).start()
    threading.Thread(target=runMCLLoop, args=(simWorld, finished)).start()
    threading.Thread(target=WaitForChar, args=(finished, '[Qq]')).start()
    threading.Thread(target=runCozmoMainLoop, args=(simWorld, finished)).start()
    # running the plot loop in a thread is not thread-safe because matplotlib
    # uses tkinter, which in turn has a threading quirk that makes it
    # non-thread-safe outside the python main program.
    # See https://stackoverflow.com/questions/14694408/runtimeerror-main-thread-is-not-in-main-loop

    # threading.Thread(target=runPlotLoop, args=(simWorld,finished)).start()
    runPlotLoop(simWorld, finished)


# NOTE: this code allows to specify the initial position of Cozmo on the map
simWorld = CozmoSimWorld(m, Frame2D.fromXYA(500, 100, math.pi))
# simWorld = CozmoSimWorld(m,Frame2D.fromXYA(200,350,-3.1416/2))

cozmo_program(simWorld)
