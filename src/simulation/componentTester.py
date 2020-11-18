from src.simulation.Track import loadTrack, plotTrack
import matplotlib.pyplot as plt
import numpy as np

blergh = np.array([[1],[2],[3]])
print(blergh)
m = loadTrack("Oval.csv")

#plt.ion()
plt.show()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)

ax.set_xlim(m.minX, m.maxX)
ax.set_ylim(m.minY, m.maxY)

plotTrack(ax, m, False)
plt.show()
