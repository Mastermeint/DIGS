#!/usr/bin/python3

# author: Meinte Ringia

# using NetworkX to implement dotted interval graphs as 
# sequence interval graphs

# documentation on NetworkX is on
# https://networkx.github.io/


import networkx as nx


G=nx.Graph()

# source for computing fast prime list
# http://stackoverflow.com/questions/16004407/a-fast-prime-number-sieve-in-python
def sieve_for_primes_to(n):
    size = n//2
    sieve = [1]*size
    limit = int(n**0.5)
    for i in range(1,limit):
        if sieve[i]:
            val = 2*i+1
            tmp = ((size-1) - i)//val 
            sieve[i+val::val] = [0]*tmp
    return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i>0]


# G.add_node(1)
# G.add_nodes_from([2,3])

# H=nx.path_graph(10)
# G.add_nodes_from(H)
# G.add_node(H)


# every DIG vertex is represented as a tuple:
#           (offset, jumpsize, stepNumber)
# TODO: Implement offset and stepnumber
def convert_to_dig( G ):
    "converts a graph to a dotted interval graph"
    listOfNodes = G.nodes()
    listOfEdges = G.edges()

    N = length(listOfNodes)
    NLisPrime = sieve_for_primes_to(N)
    
    # initialize DIG vertex
    for it in range(N):
        DIG = []
        p0 = NLisPrime(it)
        jumpsize = p0
        offset = 0
        stepNumber = 0

        for it2 in range(it):
            if (G.has_edge(it,it2) || G.has_edge(it2,it)):
                pm = NLisPrime(it2)
                jumpsize = jumpsize * pm
                #offset = offset * pm
        #offset = (offset + 1) * p0

        DIG.append((offset, jumpsize, stepNumber))
    return DIG
