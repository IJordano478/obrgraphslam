#!/usr/bin/env python3

'''

'''
from matplotlib import pyplot as plt

import matplotlib.patches as patches
import numpy as np
import csv
import math


# Map storing a set of landmarks
# landmarks are stored in a dictionary mapping  cube-IDs on Frame2D (indicating their true position)
class Track:
    def __init__(self, landmarks):
        self.landmarks = landmarks
        self.minX = None
        self.minY = None
        self.maxX = None
        self.maxY = None

class Cone:
    def __init__(self):
        self.pos = np.identity(3)
        self.colour = 'red'
        self.size = None #change at a later time
        self.visible = False

    def __init__(self, pos: np.array, colour):
        self.pos = pos
        self.colour = colour
        self.size = None
        self.visible = False

#This function loads a "blergh.csv" file in. If you want to make your own then look at the format on Oval.csv and just
#change the filename from runSimTrack.py (line 15 or nearby)
def loadTrack(trackName):
    scale = 1

    coneArray = []
    with open(trackName, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            conePos = np.array([[math.cos(0), -math.sin(0), float(row[0])*scale], [math.sin(0), math.cos(0), float(row[1])*scale], [0., 0.,1.]])
            coneColour = row[2]
            coneArray.append(Cone(conePos, coneColour))


    minX = coneArray[0].pos[0,2]
    minY = coneArray[0].pos[1,2]
    maxX = coneArray[0].pos[0,2]
    maxY = coneArray[0].pos[1,2]

    for cone in coneArray:
        x = cone.pos[0,2]
        y = cone.pos[1,2]
        if(x < minX):
            minX = x
        if (x > maxX):
            maxX = x
        if (y < minY):
            minY = y
        if (y > maxY):
            maxY = y

    T = Track(coneArray)
    border = 2
    T.minX = minX - border
    T.minY = minY - border
    T.maxX = maxX + border
    T.maxY = maxY + border
    return T

#Can be used as a standalone program if you don't want a sim but want to see the track
def plotTrack(ax, t: Track, gridOn, color="blue"):
    if(gridOn == True):
        minX = t.minX
        maxX = t.maxX
        minY = t.minY
        maxY = t.maxY
        tick = 0.5
        numX = abs(maxX-minX)
        numY = abs(maxY-minY)

        for xIndex in range(0, math.ceil(numX/tick) + 1):
            x = minX + xIndex * tick
            bold = 0.8 if (xIndex - 1) % 5 == 0 else 0.4
            plt.plot([x, x], [minY, maxY], color, alpha=bold, linewidth=bold)
        for yIndex in range(0, math.ceil(numY/tick) + 1):
            y = minY + yIndex * tick
            bold = 0.8 if (yIndex - 1) % 5 == 0 else 0.4
            plt.plot([minX, maxX], [y, y], color, alpha=bold, linewidth=bold)

    for landmark in t.landmarks:
        x = landmark.pos[0,2]
        y = landmark.pos[1,2]
        size = 0.25
        if(landmark.colour.lower()=="blue"):
            colour = "blue"
        elif(landmark.colour.lower()=="yellow"):
            colour = "yellow"
        elif(landmark.colour.lower()=="orange"):
            colour = "orange"
        else:
            colour = "red"
        rect = patches.Rectangle((float(x) - size, float(y) - size), 2 * size, 2 * size, linewidth=2, edgecolor=str(colour), facecolor="none", zorder=0)
        ax.add_patch(rect)


