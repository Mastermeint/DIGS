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
import six

from fractions import gcd
from random import randint
from functools import reduce
from itertools import combinations
from copy import deepcopy


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
        self.graph = {}
        self.dig_type = dtype
        self.vertices = {}
        if type(vertices) == list:
            count = 0
            for vertex in vertices:
                vertex = {"offset": vertex[0], "jump": vertex[1], "steps": vertex[2]}
                self.vertices.update({count: vertex})
                count += 1
        elif type(vertices) == dict:
            if type(six.next(six.itervalues(vertices))) == dict:
                self.vertices = vertices
            else:
                self.vertices = {i:vertices[i] for i in range(len(vertices))}
        else:
            raise TypeError ("vertices not of type dict or list")
        try:
            self.max_jump = max( x['jump'] for k,x in self.vertices.items() )
            self.start = min( x['offset'] for k,x in self.vertices.items() )
            self.finish = max( x['offset'] + (x['jump']*x['steps']) for k,x in self.vertices.items() )
        except:
            self.max_jump = 0
            self.start = start
            self.finish = finish
        self.node = self.vertices


    def __iter__(self):
        for obj in self.vertices:
            yield obj

    def __len__(self):
        """defines length of object by number of vertices"""
        return len(self.vertices)

    def __contains__(self, n):
        try:
            return n in self.node
        except TypeError:
            return False

    def name(self):
        return self.graph.get('name', '')

    def name(self, s):
        self.graph['name']=s

    def __str__(self):
        return self.name

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

    def has_edge(self, i, j):
        """returns True if node i is connected to node j"""
        if i == j:
            return False

        offset1 = self.vertices[i]['offset']
        offset2 = self.vertices[j]['offset']

        jump1 = self.vertices[i]['jump']
        jump2 = self.vertices[j]['jump']

        steps1 = self.vertices[i]['steps']
        steps2 = self.vertices[j]['steps']

        if (jump1 == 0 and jump2 == 0):
            if (offset1 != offset2):
                return False
            else:
                return True

        if (((offset1 <= offset2 and 
            offset1 + jump1*steps1 >= offset2)
            or 
            (offset2 < offset1 and
            offset2 + jump2*steps2 >= offset1))
            and
            (offset1 - offset2) % gcd(jump1, jump2) == 0):
            return True
        return False

    def edges_iter(self, nbunch=None):
        """iterator over edges"""
        if nbunch is None:
            for i, j in combinations(self.vertices, 2):
                if self.has_edge(i,j):
                    yield (i,j)
        elif type(nbunch == int) or type(nbunch == str):
            for i in self.nodes():
                if self.has_edge(nbunch, i):
                    yield (i,nbunch)
        else:
            for i in nbunch:
                for j in self.nodes():
                    if self.has_edge(i,j):
                        yield (i,j)

    def edges(self, nbunch=None):
        """Return list of edges"""
        return [i for i in self.edges_iter(nbunch)]

    def number_of_edges(self):
        """returns number of edges"""
        return len(self.edges())

    def __getitem__(self, index):
        item = {}
        for v1, v2 in self.edges(index):
                if v1 == index:
                    item.update({v2:{}})
                elif v2 == index:
                    item.update({v1:{}})
                else:
                    raise ValueError("something went wrong in getitem")
        return item

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
            elif (type(label) == str) or (type(label) == int):
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

    def is_directed(self):
        """returns False since DIGs are no directedgraphs"""
        return False

    def is_multigraph(self):
        """returns False since DIGs are no multigraphs"""
        return False

    def copy(self):
        """returns a copy of the object"""
        return deepcopy(self)
        
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
                yield j

    def neighbors(self, i):
        """returns al list of all nodes directly connected to node i """
        return [neigh for neigh in self.neighbors_iter(i)]

    def degree(self, node):
        return len(self.neighbors(node))

    def degree_iter(self, node):
        if type(node) == str or type(node) == int:
            for neighbors in self.neighbors_iter(node):
                yield (node, self.degree(neighbors))
        elif type(node) == list:
            for vertex in node:
                for neighbors in self.neighbors_iter(vertex):
                    yield (vertex, self.degree(neighbors))
        else:
            raise TypeError("node of wrong type")

    def subgraph(self, nodes):
        """Returns a subgraph consisting of the list of nodes given """
        subgraph = {}
        if type(nodes) == str or type(nodes) == int:
            subgraph[nodes] = self.vertices[nodes]
        elif type(nodes) == list:
            for node in nodes:
                subgraph[node] = self.vertices[node]
        else:
            raise TypeError("nodes not of type string, int or list")
        return DIGd(subgraph)

    # All functions that are needed solely for DIGs

    def get_intvals_on_dot(self, dot):
        intvals = []
        for key, intval in six.iteritems(self.vertices):
            offset = intval['offset']
            steps = intval['steps']
            jump = intval['jump']

            if (offset <= dot) and (dot <= offset + steps*jump):
                if ((dot-offset) % jump) == 0:
                    intvals.append(key)
        return intvals

# def greedy_color(D):
#     """Greedy coloring algorithm by Vladimir Yanovsky """
#     coloring = dict.fromkeys(D.vertices.keys(),[])
#     coloring_end_x = []
#     coloring_passing_y = []
#     coloring_around_z = []
#     for i in range(D.start, D.finish+2):
#         color_not_used_in_i = colors - prev colors
#         
#         for jump = range(D.max_jump):
#             color_passing_and_end = [x for x in coloring_end_x or coloring_passing_y]
#             color_not_used_in_i_with_d = [x in coloring_around_z and x not in color_passing_and_end]
# 
#         for p in Pdi:
#             if len(color_not_used_in_i_with_d):
#                 color(p) = pop(color_not_used_in_i)
#             elif len(color_not_used_in_i):
#                 color(p) = pop(color_not_used_in_i_with_d)
#             else:
#                 c = newColor
#                 C = C.append(c)
#                 color(p) = c
#     return coloring

# create a cycle in DIG2 of length n
# assume a cycle is always of length larger than 0
def create_cycle(n):
    """Creates a cycle of length n in DIG2 structure"""
    if n == 0: return DIGd([], None)
    vertices = [(0,1,1)]
    for num_vertex in range(n-2):
        vertices.append((num_vertex,2,1)) 
    vertices.append((n-2,1,1))
    return DIGd(vertices, "cycle")

# create a complete bipartite graph of size k in DIGk, k>0
def create_complete_bipartite(k):
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
def create_random_dig(intval_end, max_jump, num_vertices):
    """returns a random DIGd graph

    Keywords:
    tup_int_val -- expects a 2-tuple "(start, finish)" where DIG is created
    max_jump    -- maximum jump of every interval
    num_vertices-- number of vertices (i.e. intervals) to be created within interval
    """
    vertices = {}

    for i in range(num_vertices):
        offset = randint(0, intval_end)
        jump = randint(0, min(max_jump,intval_end - offset))
        steps = randint(0, (intval_end - offset) // max(max_jump,1))
        
        vertices[i] = {"offset": offset,
                        "jump": jump,
                        "steps": steps} 

    return DIGd(vertices)


# Find least common divisor for reduce function
def lcm(numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

# use observations from Optimization problems in DIGs to make sure
# that the interval of the DIG is in 4*n*lcm(d)

#vertex: offset, jumpsize, steps

#checks if there are no start of finish points in [i,i+2l(d) -1]
def reduce_dig(D):
    """reduces the DIGd interval such that the interval is no longer than 4*n*lcm(D)
    where n is the number of vertices and lcm is the least common multiplier of all
    jumps of the intervals in D."""
    vertices = D.vertices
    ld = lcm([subdic['jump'] for k, subdic in vertices.items()])
    reduce_count = 0
    vertex_num = len(D.vertices)
    
    for i in range(D.start, D.finish - 2*ld + 1):
        start_finish_in_intval = False
        dic_of_reduc_vert = {}

        # check if a start or end point is in the interval
        for key, vertex in vertices.items():
            dic_of_reduc_vert[key] = {1}
            offset = vertex['offset']
            jump = vertex['jump']
            steps = vertex['steps']

            start_vert = offset 
            end_vert = offset + jump*steps

            if (i <= start_vert <= (i + 2*ld -1)):
                start_finish_in_intval = True
                break
            elif (i <= (end_vert) <= (i+2*ld -1)):
                start_finish_in_intval = True
                break
            # if interval is not in [i,i+2*ld -1] then do nothing
            elif (((start_vert < i) and (end_vert < i))
            or ((start_vert > i+2*ld -1) and (end_vert > i+2*ld))):
                dic_of_reduc_vert[key] = 0
                
        if start_finish_in_intval == True: 
            continue
        if (D.finish - D.start) - (reduce_count*ld + i) < 0: break

        reduce_count += 1
        vertices = reduce_intval(vertices, dic_of_reduc_vert, i, ld)
    return DIGd(vertices)    

# reduce current L to L(i) (see D.Hermelin et al.)
def reduce_intval(vertices, dic_of_reduc_vert, i, ld):
    for key, change_vertex in dic_of_reduc_vert.items():
        if change_vertex:
            offset = vertices[key]['offset']
            jump = vertices[key]['jump']
            steps = vertices[key]['steps']

            vertices[key] = {'offset': offset, 'jump': jump, 'steps': steps-(ld//jump)}
    return vertices

def sample():
    D = DIGd([(1,2,20),(0,2,20),(1,1,4)])
    return D

import os
import numpy as np

if __name__ == "__main__":
    try:
        os.remove("workfile.txt")
    except:
        pass

    f = open('workfile.txt', 'w+')
    f.write('(intval, jump) \n')

    intvals = (10, 50)
    exist_intval = 10
    maxjump = 3
    repetitions = 10000
    count = 0

    for intvalnum in range(intvals[0], intvals[1]):
        print("interval: " + str(intvalnum))
        for jump in range(2,maxjump):
            av_dens = 0
            for repeat in range(repetitions):
                d = create_random_dig(exist_intval, jump, intvalnum)
                av_dens += nx.density(d)
            av_dens = av_dens/repeat
            f.write('{:10.10f} {count}\n'.format(av_dens, count=count))
            count += 1
    f.close()
    # d = create_random_dig((0,100), 3, 30) 
    # print("density of d: " + str(nx.density(d)))
    # print("coloring of d: ")
    # print(nx.coloring.greedy_color(d))
    # print("maximal_matching of d: ")
    # print(nx.matching.maximal_matching(d))



