from src.simulation.Track import loadTrack, plotTrack
import matplotlib.pyplot as plt

m = loadTrack("Oval.csv")

#plt.ion()
plt.show()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)

ax.set_xlim(m.grid.minX(), m.grid.maxX())
ax.set_ylim(m.grid.minY(), m.grid.maxY())

plotTrack(ax, m)
plt.show()
