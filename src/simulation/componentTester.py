from src.simulation.Track import loadTrack, plotTrack
from src.simulation.variables import *
import matplotlib.pyplot as plt
import numpy as np

blergh = np.array([[1],[2],[3]])

xy = np.array([[20, -20, -40, -40, -25, -25, -40, -40, -20,    20,  40,  40,  25, 25, 40, 40, 20],
                   [150,  150,  90,  60,  10, -40, -70, -90, -130, -130, -90, -70, -40, 10, 60, 90, 150]])
rotation = radiansTo2DMatAngle(-math.pi/2)
print(np.transpose(np.matmul(np.transpose(xy),rotation)))
m = loadTrack("Oval.csv")

#plt.ion()
plt.show()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)

ax.set_xlim(m.minX, m.maxX)
ax.set_ylim(m.minY, m.maxY)

plotTrack(ax, m, False)
plt.show()
