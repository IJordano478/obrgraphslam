from src.simulation.Track import loadTrack, plotTrack
from src.simulation.variables import *
from src.graphslam import *
import matplotlib.pyplot as plt
import numpy as np
import csv
from numpy import newaxis


class Graph():
    nodes = []
    recentNode = None

    'Node subclass, as Nodes are only used by Graph'
    class Node():
        def __init__(self, pose=None):
            """
            Attributes:
                _pose           3x3 transformation matrix of robot xya
                _nextNodes      list of directly connected Nodes
                _nextEdge       list of directly connected Edges as the relative transformation between 2 nodes
            """
            self._pose = pose
            self._nextNodes = []
            self._nextEdges = []

        def getPose(self):
            return self._pose

        def getNextNodes(self):
            return self._nextNodes

        def getNextEdges(self):
            return self._nextEdges

        def setPose(self, nPose):
            self._pose = nPose

        def setNextNodes(self, updatedNodes):
            self._nextNodes = updatedNodes

        def setNextEdges(self, updatedEdges):
            self._nextEdges = updatedEdges


    def getAllNodes(self):
        return self.nodes

    def addNode(self, newNode, connectTo=None, edge=None):
        """
        :param newNode: The new node to be added as transformation mat
        :param connectTo: The previous node
        :param edge: The relative transformation to get from the original node to the new node
        :return:
        """
        #The node needs to created without any previous connections
        if (connectTo is None) & (edge is None):
            newNode = self.Node(newNode)
            self.nodes.append(newNode)
            return

        #The new node needs to be connected to a previous node
        #elif connectTo is not None & edge is None:
        else:
            if edge is None:
                relative = np.matmul(np.linalg.inv(connectTo), newNode)
            for node in self.nodes:
                if np.array_equal(node.getPose(),connectTo):
                    #origin node found, make connection
                    self.nodes.append(self.Node(newNode))
                    updatedNextNodes = node.getNextNodes() + [self.nodes[-1]]
                    updatedNextEdges = node.getNextEdges() + [relative]
                    node.setNextNodes(updatedNextNodes)
                    node.setNextEdges(updatedNextEdges)
                    return

    def deleteNode(self):
        # TODO delete a node based on its pos
        return

    def updateNode(self):
        # TODO update a node at pos X
        return

    def listAllNodes(self):
        # TODO iterate through list and return XYA
        return

    def plotGraph(self):
        # TODO plot entire graph on malplot
        return



#
'''
xyab = transformationMat(1, 2, 0)
xya1 = transformationMat(2, 2, 0)
xya2 = transformationMat(0, 0, math.pi / 2)
xya3 = transformationMat(0, 1, math.pi / 2)


graph = Graph()
graph.addNode(xyab)
graph.addNode(xya1,xyab)
#basepos.addNode(xya1)
#basepos.addNode(xya2)

#print(basepos.getConnectionAtI(0))

# =====Testing_Stage=====
# More than welcome to use this file to test functions and bits separately
blergh = np.array([[1], [2], [3]])

xy = np.array([[20, -20, -40, -40, -25, -25, -40, -40, -20, 20, 40, 40, 25, 25, 40, 40, 20],
               [150, 150, 90, 60, 10, -40, -70, -90, -130, -130, -90, -70, -40, 10, 60, 90, 150]])
rotation = radiansTo2DMatAngle(-math.pi / 2)
print(np.transpose(np.matmul(np.transpose(xy), rotation)))
m = loadTrack("Oval.csv")

# plt.ion()
#plt.show()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)
x = []
y = []
all = []
with open("mean.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:

        all.append(float(row[0]))

for i in range(0, len(all), 2):
    x.append(all[i])
    y.append(all[i+1])
print(x)
print(y)
plt.scatter(x, y, marker="x")
plt.show()
'''


#########SHOW ONLY CONE POSITIONS FROM A MEAN VECTOR############
x = np.empty((0,1))
y = np.empty((0,1))
all = np.empty((0,1))
with open("mean.csv", 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:
        row = float(row[0])
        row = np.array([[row]])
        all = np.concatenate((all, row), axis=0)

plt.cla()  # Clear current plot

# Stops the visualization with the esc key.
plt.gcf().canvas.mpl_connect(
    'key_release_event', lambda event:
    [exit(0) if event.key == 'escape' else None])

for i in range(3, all.shape[0], 2):
    print(all[i, :])
    if x.shape[0] == 0:
        x = np.array([all[i, :]])
        y = np.array([all[i+1, :]])
    else:
        x = np.concatenate((x, np.array([all[i,:]])))
        y = np.concatenate((y, np.array([all[i+1, :]])))
coneplt = plt.scatter(x, y, marker="x", color="r", label="SEIF Cones")

plt.legend()
plt.title('Visualizing Measurement State')
plt.xlabel('X distance (m)')
plt.ylabel('Y distance (m)')
plt.axis("equal")
plt.grid(True)
plt.pause(100)



'''
#########SHOW LINKS GIVEN OMEGA AND XI############
active = False
connections = True
holdprogram = True
xi = np.loadtxt(open("xi.csv", "rb"), delimiter=",")
xi = np.expand_dims((xi), axis=1)
#temp = np.genfromtxt(open("xi.csv", "rb"), delimiter=',')
#xi = [np.array(arr) for arr in temp[:, np.newaxis]]
omega = np.loadtxt(open("omega.csv", "rb"), delimiter=",")

if type(omega) == Omega2:
    omega = omega.omegaMatrix

if type(xi) == Xi:
    xi = xi.xiVector

mean = np.linalg.pinv(omega) @ xi

x = np.array([])
y = np.array([])

for i in range(3, mean.shape[0], 2):
    x = np.append(x, mean[i, 0])
    y = np.append(y, mean[i + 1, 0])

# print(x)
# print(y)
fig, ax = plt.subplots(figsize=(10, 10))
axes = plt.gca()
axes.set_xlim([0, 10])
axes.set_ylim([0, 18])

if connections:
    for r in range(3, omega.shape[0], 2):
        for c in range(3, omega.shape[0], 2):
            if not np.allclose(omega[r:r + 2, c:c + 2], np.zeros((2, 2))):
                plt.plot([mean[r, 0], mean[c, 0]], [mean[r + 1, 0], mean[c + 1, 0]], color="Black", lw=1)

if active:
    for i in range(3, omega.shape[0], 2):
        if not np.allclose(omega[0:3, i:i + 2], np.zeros((3, 2))):
            plt.plot([mean[0, 0], mean[i, 0]], [mean[1, 0], mean[i + 1, 0]], color="Red")

ax.scatter(x, y, marker="*", label="Cone Positons")
ax.scatter(mean[0, 0], mean[1, 0], marker="o")
plt.legend()
plt.title('Omega Graph Links')
plt.xlabel('X distance (m)')
plt.ylabel('Y distance (m)')

if holdprogram:
    plt.show()
else:
    plt.show(block=False)
'''
