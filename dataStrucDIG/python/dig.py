#!/usr/bin/python3

# author: Meinte Ringia

# using NetworkX to implement dotted interval graphs as 
# sequence interval graphs

# documentation on NetworkX is on
# https://networkx.github.io/


import networkx as nx


G=nx.Graph()

# G.add_node(1)
# G.add_nodes_from([2,3])

# H=nx.path_graph(10)
# G.add_nodes_from(H)
# G.add_node(H)

def convert_to_dig( G ):
    "converts a graph to a dotted interval graph"
    listOfNodes = G.nodes()
    listOfEdges = G.edges()

