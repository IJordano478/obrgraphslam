#!/usr/bin/env python3

# Copyright (c) 2019 Matthias Rolf, Oxford Brookes University

'''

'''
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import csv
import math
#import cozmo

from src.simulation.frame2d import Frame2D


class Coord2D:
    def __init__(self, xp: float, yp: float):
        self.x = xp
        self.y = yp

    def __str__(self):
        return "[x=" + str(self.x) + ",y=" + str(self.y) + "]"


class Coord2DGrid:
    def __init__(self, xp: int, yp: int):
        self.x = xp
        self.y = yp

    def __str__(self):
        return "[index-x=" + str(self.x) + ",index-y=" + str(self.y) + "]"


class OccupancyGrid:
    FREE = 0
    OCCUPIED = 1

    def __init__(self, start: Coord2D, stepSize, sizeX, sizeY):
        self.gridStart = start
        self.gridStepSize = stepSize
        self.gridSizeX = sizeX
        self.gridSizeY = sizeY
        self.gridData = np.zeros((sizeX, sizeY), int)

    def validateIndex(self, c: Coord2D):
        if c.x < 0 or self.gridSizeX <= c.x:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds.")
        if c.y < 0 or self.gridSizeY <= c.y:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds.")

    def validateIndexStop(self, c: Coord2D):
        if c.x < -1 or self.gridSizeX < c.x:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds for index stop.")
        if c.y < -1 or self.gridSizeY < c.y:
            raise Exception("OccupancyGrid coordinate ", str(c), " is out of bounds for index stop.")

    def float2grid(self, c: Coord2D):
        xIndex = round((c.x - self.gridStart.x) / self.gridStepSize)
        yIndex = round((c.y - self.gridStart.y) / self.gridStepSize)
        ci = Coord2DGrid(xIndex, yIndex)
        self.validateIndex(ci)
        return ci

    def grid2float(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        x = self.gridStart.x + ci.x * self.gridStepSize
        y = self.gridStart.y + ci.y * self.gridStepSize
        return Coord2D(x, y)

    def isFreeGrid(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        return self.gridData[int(ci.x), int(ci.y)] == self.FREE

    def isFree(self, c: Coord2D):
        return self.isFreeGrid(self.float2grid(c))

    def isOccupiedGrid(self, ci: Coord2DGrid):
        self.validateIndex(ci)
        return self.gridData[int(ci.x), int(ci.y)] == self.OCCUPIED

    def isOccupied(self, c: Coord2D):
        return self.isOccupiedGrid(self.float2grid(c))

    def setFree(self, start: Coord2DGrid, end: Coord2DGrid):
        self.validateIndex(start)
        self.validateIndexStop(end)
        for x in range(start.x, end.x):
            for y in range(start.y, end.y):
                self.gridData[x, y] = self.FREE

    def setOccupied(self, start: Coord2DGrid, end: Coord2DGrid):
        self.validateIndex(start)
        self.validateIndexStop(end)
        for x in range(start.x, end.x):
            for y in range(start.y, end.y):
                self.gridData[x, y] = self.OCCUPIED

    def minX(self):
        return self.gridStart.x - 0.5 * self.gridStepSize

    def minY(self):
        return self.gridStart.y - 0.5 * self.gridStepSize

    def maxX(self):
        return self.gridStart.x + (self.gridSizeX - 0.5) * self.gridStepSize

    def maxY(self):
        return self.gridStart.y + (self.gridSizeY - 0.5) * self.gridStepSize

    def __str__(self):
        g = ""
        for x in range(0, self.gridSizeX):
            line = ""
            for y in range(0, self.gridSizeY):
                if self.gridData[x, y] == self.FREE:
                    line = line + ".. "
                elif self.gridData[x, y] == self.OCCUPIED:
                    line = line + "XX "
            g = g + line + "\n"
        return g


# Map storing an occupancy grip and a set of landmarks
# landmarks are stored in a dictionary mapping  cube-IDs on Frame2D (indicating their true position)
class Track:
    #def __init__(self, grid, landmarks, targets=None):
    def __init__(self, grid, landmarks):
        self.grid = grid
        self.landmarks = landmarks
        #self.targets = targets


def loadTrack(trackName):

    coneArray = []
    with open('Oval.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            coneArray.append(row)

    minX = float(coneArray[0][0])
    minY = float(coneArray[0][1])
    maxX = float(coneArray[0][0])
    maxY = float(coneArray[0][1])

    for cone in coneArray:
        x = float(cone[0])
        y = float(cone[1])
        if(x < minX):
            minX = x
        if (x > maxX):
            maxX = x
        if (y < minY):
            minY = y
        if (y > maxY):
            maxY = y

    minX = math.ceil(minX)
    minY = math.ceil(minY)
    maxX = math.ceil(maxX)
    maxY = math.ceil(maxY)
    stepSize = 1

    grid = OccupancyGrid(Coord2D(minX, minY), stepSize, int((maxX-minX)/stepSize)+1, int((maxY-minY)/stepSize)+1)
    #grid.setOccupied(Coord2DGrid(0, 0), Coord2DGrid(sizeX, sizeY))
    #grid.setFree(Coord2DGrid(1, 1), Coord2DGrid(sizeX - 1, sizeY - 1))
    #grid.setOccupied(Coord2DGrid(16, 21), Coord2DGrid(sizeX, 23))

    return Track(grid, coneArray)


def plotTrack(ax, m: Track, color="blue"):
    grid = m.grid
    minX = grid.minX()
    maxX = grid.maxX()
    minY = grid.minY()
    maxY = grid.maxY()
    tick = grid.gridStepSize
    numX = grid.gridSizeX
    numY = grid.gridSizeY
    for xIndex in range(0, numX + 1):
        x = minX + xIndex * tick
        bold = 0.8 if (xIndex - 1) % 5 == 0 else 0.4
        plt.plot([x, x], [minY, maxY], color, alpha=bold, linewidth=bold)
    for yIndex in range(0, numY + 1):
        y = minY + yIndex * tick
        bold = 0.8 if (yIndex - 1) % 5 == 0 else 0.4
        plt.plot([minX, maxX], [y, y], color, alpha=bold, linewidth=bold)

    for landmark in m.landmarks:
        x = landmark[0]
        y = landmark[1]
        size = 0.25
        if(landmark[2].lower()=="blue"):
            colour = "blue"
        elif(landmark[2].lower()=="yellow"):
            colour = "yellow"
        elif(landmark[2].lower()=="orange"):
            colour = "orange"
        else:
            colour = "red"
        rect = patches.Rectangle((float(x) - size, float(y) - size), 2 * size, 2 * size, linewidth=2, edgecolor=str(colour), facecolor="none", zorder=0)
        ax.add_patch(rect)


