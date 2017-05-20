import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations 
from fractions import gcd
from random import randint

#vertex: offset, jumpsize, steps
class DIGd:
    def __init__(self, vertices = [], dtype = None):
        try:
            self.max_jump = max(vertices, key = lambda x:x[1])[1]
        except ValueError:
            self.max_jump = 0
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
    D = DIGd([(0,1,1)], "cycle")
    if n == 1: return D
    for num_vertex in range(n-2):
        D.max_jump = 2
        D.vertices.append((num_vertex,2,1)) 
    D.vertices.append((n-2,1,1))
    return D

# create a complete bipartite graph of size k in DIGk, k>0
def dig_complete_bipartite(k):
    D = DIGd([], "cbipartite")
    D.max_jump = k
    for i in range(k):
        D.vertices.append((i,k,k-1))
    for j in range(k):
        D.vertices.append((j*k,1,k-1))
    return D

def plot_bipartite_graph(G):
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
    vertices = []
    for i in range(num_vertices):
        offset = randint(tup_int_val[0], tup_int_val[1])
        jump = randint(0, min(max_jump,tup_int_val[1]-offset))
        steps = randint(0, (tup_int_val[1]-offset)//max_jump)
        vertices.append((offset, jump, steps))
    return DIGd(vertices)


