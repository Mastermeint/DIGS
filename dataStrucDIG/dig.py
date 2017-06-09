#!/usr/bin/env python3
#
# author: Meinte 


"""
This module represents graphs as Dotted Interval Graphs.
It is combined with module networkX 

Classes:
    DIGd

Functions:
    dig_cycle
    dig_complete_bipartite
    plot_bipartite_graph
    create_random_dig
    reduce_dig
"""


import networkx as nx
import matplotlib.pyplot as plt

from fractions import gcd
from random import randint
from functools import reduce
from itertools import combinations


# convert vertices to dictionary

#vertex: offset, jumpsize, steps
class DIGd:
    """Dotted interval Graph structure
   
    Description:
    Represents a dotted interval graphs with vertices

    Keywords:
    max_jump    -- maximam of jumps of all dotted intervals (default 0)
    start       -- value of position of lowest dot (default 0)
    finish      -- position of last dot of all intervals (default 0)
    dig_type    -- type of the DIG (default None)
    vertices    -- list of all vertices as three tuple (offset, jumpsize, steps) (default [])
    
    Functions:
    dig_to_networkX()   -- converts a DIG to networkX structure in O(n^2) resp. to the vertices n
    nodes_iter()
    number_of_nodes()
    edges_iter()
    edges()
    number_of_edges()
    size()
    non_edges()
    adjacency_iter()
    clear()
    add_nodes_from(nodes)
    add_edges_from(nodes)
    remove_nodes_from(nodes)
    remove_edges_from(node)
    has_edge(i,j)
    is_directed()
    is_multigraph()
    copy()
    dig_to_networkX()
    neighbors_iter(node)
    neighbors(node)
    degree(node)
    degree_iter(node)
    subgraph(nodes)
    """
    def __init__(self, vertices = [], dtype = None, start = 0, finish = 0):
        self.name = 'DIG'
        try:
            self.max_jump = max(vertices, key = lambda x:x[1])[1]
            self.start = min( [x[0] for x in vertices]  )
            self.finish = max([x[0]+(x[1]*x[2]) for x in vertices])
        except:
            self.max_jump = 0
            self.start = start
            self.finish = finish
        self.dig_type = dtype
        self.vertices = {}
        if type(vertices) == list:
            count = 0
            for vertex in vertices:
                vertex = {"offset": vertex[0], "jump": vertex[1], "steps": vertex[2]}
                self.vertices.update({count: vertex})
                count += 1
        elif type(vertices) == dict:
            self.vertices = {i:vertices[i] for i in range(len(vertices))}
        else:
            raise TypeError ("vertices not of type dict or list")

    # Beetje hacky, maar het werkt
    def __getitem__(self, index):
        return list(self.vertices.keys())[index]

    def __iter__(self):
        for key in self.vertices:
            yield key

    def __len__(self):
        """defines length of object by number of vertices"""
        return len(self.vertices)

    def __contains__(self, n):
        try:
            return n in self.node
        except TypeError:
            return False

    def nodes_iter(self):
        """iterator over nodes"""
        for i in self.vertices.keys():
            yield i

    def nodes(self):
        """Return list of nodes"""
        return [i for i in self.nodes_iter()]

    def number_of_nodes(self):
        """Returns number of nodes"""
        return len(self.nodes())

    def edges_iter(self):
        """iterator over edges"""
        for i in range(len(self)):
            for j in range(i+1, len(self)):
                if self.has_edge(i,j):
                    yield (i, j)

    def edges(self):
        """Return list of edges"""
        return [i for i in self.edges_iter()]

    def number_of_edges(self):
        """returns number of edges"""
        return len(self.edges())

    def size(self):
        return self.number_of_edges()

    def non_edges(self):
        """Returns list of all non-existant edges"""
        all_edges = [(i,j) for i, j in combinations(self.nodes(), 2)]
        return [edge for edge in all_edges if edge not in self.edges()]

    def adjacency_iter(self):
        """Return an iterator of (node, adjacency dict) tuples for all nodes."""
        for i in range(len(self)):
            yield (i, {j: {} for j in range(len(self)) if self.has_edge(i,j) and j != i})

    def clear(self):
        """sets all properties to default"""
        self.vertices = {}
        self.max_jump = 0
        self.start = 0
        self.finish = 0
        self.dig_type = None

    def add_node(self, label):
        """Adds a single node,

        Arguments:
            label --    can be a string, then a disjunct node will be added
                        can be a dictionary, then a dotted interval will be added"""
        try:
            if all (k in label for k in ("offset", "jump","steps")):
                if "name" in label:
                    self.vertices[label["name"]] = {"offset": label["offset"],
                                                    "jump": label["jump"],
                                                    "steps": label["steps"]}
                else:
                    raise TypeError ("name not in dictionary")
            elif type(label) == str:
                self.vertices[label] = {"offset":self.finish+1,
                                        "jump" : 0,
                                        "steps" : 0}
            self.finish += 1
        except:
            raise TypeError ("node not of format dict with keys offset, jump, steps, "
                              "or not of gormat string")
                    
    def add_nodes_from(self, nodes):
        """adds singletons after finish"""
        for node in nodes:
            self.add_node(node)

    def add_edges_from(self, lis):
        """defined with pass"""
        raise NotImplementedError

    # remove all vertices by index from self.vertices
    def remove_nodes_from(self, nodes):
        """removes nodes listed"""
        for i in nodes:
            del self.vertices[i]
        
    def remove_edges_from(self):
        """defined with pass"""
        raise NotImplementedError

    def has_edge(self, i, j):
        """returns True if node i is connected to node j"""
        offset1 = self[i]['offset']
        offset2 = self[j]['offset']

        jump1 = self[i]['jump']
        jump2 = self[j]['jump']

        steps1 = self[i]['steps']
        steps2 = self[j]['steps']
        
        vertex1 = self.vertices[i]
        vertex2 = self.vertices[j]

        if (((offset1 <= offset2 and 
            offset1 + jump1*steps1 >= offset2)
            or 
            (offset2 < offset1 and
            offset2 + jump2*steps2 >= offset1))
            and
            (offset1 - offset2) % gcd(jump1, jump2) == 0):
            return True
        return False

    def is_directed(self):
        """returns False since DIGs are no directedgraphs"""
        return False

    def is_multigraph(self):
        """returns False since DIGs are no multigraphs"""
        return False

    # not sure if this is what the copy function should look like
    def copy(self):
        """returns a copy of the object"""
        return DIGd(self.vertices)
        
    def dig_to_networkX(self):
        """converts DIGd struc to networkx struc"""
        G = nx.Graph()
        for i in range(len(self)):
            G.add_node(self[i])
            for j in range(i+1, len(self)):
                if self.has_edge(i,j):
                    G.add_edge(self[i],self[j])
        return G

    def neighbors_iter(self, i):
        """returns al list of all nodes directly connected to node i """
        neighbors = []
        for j in self.nodes():
            if self.has_edge(i,j) and i != j:
                yield (i, j)

    def neighbors(self, i):
        """returns al list of all nodes directly connected to node i """
        return [neigh for neigh in self.neighbors_iter(i)]

    def degree(self, node):
        return len(self.neighbors)

    def degree_iter(self, node):
        for n in self.nodes():
            yield (n, neighbors)

    def subgraph(self, nodes):
        """Returns a subgraph consisting of the list of nodes given """
        return DIGd([i for j, i in enumerate(self.vertices) if j in nodes])

#     def nbunch_iter(self, nbunch):
#         for node in nbunch:
#             yield node
#         def bunch_iter
#         try:
#                             for n in nlist:
#                                                     if n in adj:
#                                                                                yield n
                
# create a cycle in DIG2 of length n
# assume a cycle is always of length larger than 0
def dig_cycle(n):
    """Creates a cycle of length n in DIG2 structure"""
    if n == 0: return DIGd([], None)
    vertices = [(0,1,1)]
    for num_vertex in range(n-2):
        vertices.append((num_vertex,2,1)) 
    vertices.append((n-2,1,1))
    return DIGd(vertices, "cycle")

# create a complete bipartite graph of size k in DIGk, k>0
def dig_complete_bipartite(k):
    """Creates a complete bipartite graph of size k in DIGk with k > 0"""
    vertices = []

    for i in range(k):
        vertices.append((i, k, k-1))
    for j in range(k):
        vertices.append((j*k, 1, k-1))

    return DIGd(vertices, "cbipartite")

# plot a bipartite graph G, where G has a networkX struc
def plot_bipartite_graph(G):
    """ plots a bipartite graph from a networkX structure """
    # Separate by group
    l, r = nx.bipartite.sets(G)
    pos = {}

    # Update position for node from each group
    pos.update((node, (1, index)) for index, node in enumerate(l))
    pos.update((node, (2, index)) for index, node in enumerate(r))

    nx.draw(G, pos=pos)
    plt.show()

# create a random dig within interval tup_int_val 
# with maximal jump max_jump containing num_vertices vertices
def create_random_dig(tup_int_val, max_jump, num_vertices):
    """returns a random DIGd graph

    Keywords:
    tup_int_val -- expects a 2-tuple "(start, finish)" where DIG is created
    max_jump    -- maximum jump of every interval
    num_vertices-- number of vertices (i.e. intervals) to be created within interval
    """
    vertices = []
    intval_start = tup_int_val[0]
    intval_end = tup_int_val[1]

    for i in range(num_vertices):
        offset = randint(intval_start, intval_end)
        jump = randint(0, min(max_jump,intval_end - offset))
        steps = randint(0, (intval_end - offset) // max_jump)
        vertices.append((offset, jump, steps))

    return DIGd(vertices)


# Find least common divisor for reduce function
def lcm(numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

# use observations from Optimization problems in DIGs to make sure
# that the interval of the DIG is in 4*n*lcm(d)

#checks if there are no start of finish points in [i,i+2l(d) -1]
def reduce_dig(D):
    """reduces the DIGd interval such that the interval is no longer than 4*n*lcm(D)
    where n is the number of vertices and lcm is the least common multiplier of all
    jumps of the intervals in D."""
    ld = lcm([x[1] for x in D.vertices])
    reduce_count = 0
    vertex_num = len(D.vertices)
    for i in range(D.start, D.finish - 2*ld + 1):
        start_finish_in_intval = False
        list_of_reduc_vert = [1] * vertex_num

        # check if a start or end point is in the interval
        for index, vertex in enumerate(D.vertices):
            start_vert = vertex[0]
            end_vert = vertex[0] + vertex[1]*vertex[2]
            if (i <= start_vert <= (i + 2*ld -1)):
                start_finish_in_intval = True
                break
            elif (i <= (end_vert) <= (i+2*ld -1)):
                start_finish_in_intval = True
                break
            # if interval is not in [i,i+2*ld -1] then do nothing
            elif (((start_vert < i) and (end_vert < i))
            or ((start_vert > i+2*ld -1) and (end_vert > i+2*ld))):
                list_of_reduc_vert[index] = 0
                
        if start_finish_in_intval == True: 
            continue
        if (D.finish - D.start) - (reduce_count*ld + i) < 0: break

        reduce_count += 1
        D.vertices = reduce_intval(D.vertices, list_of_reduc_vert, i, ld)
    return D    

# reduce current L to L(i) (see D.Hermelin et al.)
def reduce_intval(vertices, list_of_reduc_vert, i, ld):
    for index, change_vertex in enumerate(list_of_reduc_vert):
        if change_vertex:
            offset, jump, steps = vertices[index]
            vertices[index] = (offset, jump, steps - (ld // jump))
    return vertices

def sample():
    D = DIGd([(1,2,20),(0,2,20),(1,1,4)])
    return D

if __name__ == "__main__":
    print("compiled")
    D = sample()
    d = DIGd([(1,2,3),(2,3,4),(1,1,3),(0,2,5)])
    print("graph consists of ")
    print(D.vertices)
    print(" ")
    print("vertex ", 1, "has tuple ", D[1])
    print('and neighbors ', D.neighbors(1))
    print(" ")
    print("grafenstructuur in networkx:")
    #G = D.dig_to_networkX()
    print("de vertices zijn: ", D.dig_to_networkX().nodes())
    print("met edges: ", D.dig_to_networkX().edges())
    print("++++++")
    nx.draw(d, pos=nx.spring_layout(d))



