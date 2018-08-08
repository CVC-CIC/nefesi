import networkx as nx
import matplotlib.pyplot as plt

class TreeVisualization:
    _tree = nx.Graph()
    _root = None
    _labels = {}
    def __init__(self, root = None):
        self._tree = nx.Graph()
        self._root = root
    def setRoot(self, root):
        self._root = root
    def addNode(self, node):
        if (self._tree.has_node(node)):
            raise Exception("SHIT: Node" + node + "has a duplicated Name")
        self._tree.add_node(node)
    #Adds a node and returns the real name assigned to node
    def addNodeAndReturnValidName(self, nodeName):
        flag = False
        i = 0
        while(not flag):
            try:
                self.addNode(node=nodeName)
                flag = True
            except:
                if (i%2)==0:
                    nodeName = " "+nodeName
                else:
                    nodeName = nodeName+" "
                i+=1
        return nodeName

    def addEdge(self, node1, node2, label):
        if(self._tree.has_edge(node1,node2)):
            print("You tracts to add an existing edge? Action refused")
        else:
            if(not self._tree.has_node(node1)):
                print("Node"+node1+"not added before addEdge... node added")
                self.addNode(node1)
            if(not self._tree.has_node(node2)):
                print("Node" + node2 + "not added before addEdge... node added")
                self.addNode(node2)
            self._tree.add_edge(node1, node2)
            self._labels[(node1,node2)] = label
    def getTree(self):
        positions = self.hierarchy_pos(self._tree,self._root)
        nx.draw_networkx(self._tree, positions)
        nx.draw_networkx_edge_labels(self._tree, positions, edge_labels=self._labels)
        return plt.gcf()
        #plt.savefig("graph.png", dpi=1000)

    """
    This function is extracted from https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3
    Get the layout that contains the positions of each node, to visualizate it like a tree
    """
    def hierarchy_pos(self,G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5,
                      pos=None, parent=None):
        '''If there is a cycle that is reachable from root, then this will see infinite recursion.
           G: the graph
           root: the root node of current branch
           width: horizontal space allocated for this branch - avoids overlap with other branches
           vert_gap: gap between levels of hierarchy
           vert_loc: vertical location of root
           xcenter: horizontal location of root
           pos: a dict saying where all nodes go if they have been assigned
           parent: parent of this branch.'''
        if pos == None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        neighbors = G.neighbors(root)
        if parent != None:  # this should be removed for directed graphs.
            neighbors.remove(parent)  # if directed, then parent not in neighbors.
        if len(neighbors) != 0:
            dx = width / len(neighbors)
            nextx = xcenter - width / 2 - dx / 2
            for neighbor in neighbors:
                nextx += dx
                pos = self.hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos,
                                    parent=root)
        return pos
