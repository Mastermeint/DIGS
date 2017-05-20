import networkx as nx
import itertools
from fractions import gcd

#vertex: offset, jumpsize, steps
class vertex:
    def __init__(self, offset = 0, jumpsize = 0, steps = 0):
        self.offset = offset
        self.jumpsize = jumpsize
        self.steps = steps

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
        for vertex1, vertex2 in itertools.combinations(self.vertices, 2):
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
    D = DIGd([((0,1,1))], "cycle")
    if n == 1: return D
    for num_vertex in range(n-2):
        D.max_jump = 2
        D.vertices.append((num_vertex,2,1)) 
    D.vertices.append((n-2,1,1))
    return D

#cylce1:
#[(0, 1, 1)]

#cycle2:
#[(0,1,1), (0,1,1)]

#cycle3:
#[(0,1,1), (0,2,1), (1,1,1)]

#cycle4:
#[(0,1,1), (0,2,1), (1,2,1), (2,1,1)]
    
