import networkx as nx
import matplotlib.pyplot as plt

from itertools import combinations 
from fractions import gcd
from random import randint
from functools import reduce

#vertex: offset, jumpsize, steps
class DIGd:
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

# plot a the graph G, where G had a networkX struc
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

# Find least common divisor for reduce function
def lcm(numbers):
    """Return lowest common multiple."""    
    def lcm(a, b):
        return (a * b) // gcd(a, b)
    return reduce(lcm, numbers, 1)

# use observations from Optimization problems in DIGs to make sure
# that the interval of the DIG is in 4*n*lcm(d)


def sample():
    D = DIGd([(1,2,20),(0,2,20),(1,1,4)])
    return D

#checks if there are no start of finish points in [i,i+2l(d) -1]
def reduce_dig(D):
    ld = lcm([x[1] for x in D.vertices])
    reduce_count = 0
    vertex_num = len(D.vertices)
    for i in range(D.start, D.finish - 2*ld + 1):
        start_finish_in_intval = False
        list_of_reduc_vert = [1] * vertex_num

        for index, vertex in enumerate(D.vertices):
            start_vert = vertex[0]
            end_vert = vertex[0] + vertex[1]*vertex[2]
            if (i <= start_vert <= (i + 2*ld -1)):
                start_finish_in_intval = True
                break
            elif (i <= (end_vert) <= (i+2*ld -1)):
                start_finish_in_intval = True
                break
            elif (((start_vert < i) and (end_vert < i))
            or ((start_vert > i+2*ld -1) and (end_vert > i+2*ld))):
                list_of_reduc_vert[index] = 0
                
        if start_finish_in_intval == True: 
            continue
        if (D.finish - D.start) - (reduce_count*ld + i) < 0: break
        print("continue with " + str(list_of_reduc_vert))

        #reduce current L to L(i) (see D.Hermelin et al.)
        reduce_count += 1
        D.vertices = reduce_intval(D.vertices, list_of_reduc_vert, i, ld)
    return D    

def reduce_intval(vertices, list_of_reduc_vert, i, ld):
    print("reduce for i = " + str(i))
    for index, change_vertex in enumerate(list_of_reduc_vert):
        if change_vertex:
            vertex = vertices[index]
            vertices[index] = (vertex[0], vertex[1], vertex[2] -(ld // vertex[1]))
    return vertices

# def compact_rep_dig(D):
#     compact_vertices = []
#     lcm(
#     for interval in D.vertices:
#         

#TODO:
#       implement observations from D.Hermelin et al.//Optimization problems in DIGs
#       CLEANUP_FUNCTION:   
#           sort intervals to offset
#           
