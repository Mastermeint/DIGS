import networkx as nx
import networkx.algorithms.approximation as a    
def tracefunc(frame, event, arg, indent=[0]):
    if event == "call":

        if "graph.py" in frame.f_code.co_filename:
            indent[0] += 2
            print "-" * indent[0] + "> call function", frame.f_code.co_name
    elif event == "return":
        if "graph.py" in frame.f_code.co_filename:
                print "<" + "-" * indent[0], "exit function", frame.f_code.co_name
                indent[0] -= 2
    return tracefunc

import sys
sys.settrace(tracefunc)
                                    

Graph = nx.Graph()
# Graph.add_nodes_from([1])
G = nx.complete_graph(2,Graph)
print "++++"
nx.draw(G)
# a.maximum_independent_set(Graph)
# print  "++++++"
# print a.min_weighted_vertex_cover(G)

