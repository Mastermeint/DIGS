#!/usr/bin/env python3
#
# author: Meinte 

#TODO:
#   Docstrings              -- check, should be updated
#   Python Ducktyping
#   Implement functions:
#       remove_node()
#       add_node()
#       subgraph(nodes)
#       has_edge(i,j)
#       edges()
#       neighbours(i)

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

from itertools import combinations 
from fractions import gcd
from random import randint
from functools import reduce

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
    """
    def __init__(self, vertices = [], dtype = None, start = 0, finish = 0):
        try:
            self.max_jump = max(vertices, key = lambda x:x[1])[1]
            self.start = min( [x[0] for x in vertices]  )
            self.finish = max([x[0]+(x[1]*x[2]) for x in vertices])
        except:
            self.max_jump = 0
            self.start = start
            self.finish = finish
        self.dig_type = dtype
        self.vertices = vertices
        
    def dig_to_networkX(self):
        G = nx.Graph()
        for vertex1, vertex2 in combinations(self.vertices, 2):
            G.add_node(vertex1)
            G.add_node(vertex2)
            if (((vertex1[0] <= vertex2[0] and 
                vertex1[0] + vertex1[1]*vertex1[2] >= vertex2[0])
                or 
                (vertex2[0] < vertex1[0] and 
                vertex2[0] + vertex2[1]*vertex2[2] >= vertex1[0]))
                and 
                (vertex1[0] - vertex2[0]) % gcd(vertex1[1], vertex2[1]) == 0):
                print("vertex1, vertex2: " + str(vertex1) + ", " + str(vertex2))
                G.add_edge(vertex1, vertex2)
        return G


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
    """creates a random DIGd

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

# example graph which can be reduced with reduce_dig()
def sample():
    D = DIGd([(1,2,20),(0,2,20),(1,1,4)])
    return D

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

# TODO:
# Implement Maximum independent set and vertex cover from
#    "Optimization problems in dotted interval graphs" - D.Hermelin et al.
# Implement greedy algorithm for coloring of dotted interval graphs 
#   with approximation ratio fo (2D +4)/3 from
#   "Approximation algorithm for coloring of dotted interval graphs" - V.Yanovsky

