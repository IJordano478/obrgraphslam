from src.simulation.Track import loadTrack, plotTrack
from src.simulation.variables import *
from src.graphslam import *
import matplotlib.pyplot as plt
import numpy as np
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



#print(gs_initialise([[1,0],[0,math.pi/2],[1,0]]))
#gs_linearize([1,2,3,4],[12,65,900],"c","mean")


A = np.array([[1,2,3],
          [4,5,6],
          [7,8,9]])

B = np.array([[1,2,3],
          [4,5,6],
          [7,8,9]])

C = np.array([1,2,3])

D = np.array([1,2,3])

print(np.dot(A,B))
#print(CD)
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
plt.show()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, aspect=1)

ax.set_xlim(m.minX, m.maxX)
ax.set_ylim(m.minY, m.maxY)

plotTrack(ax, m, False)
plt.show()
'''