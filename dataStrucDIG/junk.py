#!/usr/bin/python3

# author: Meinte Ringia

# using NetworkX to implement dotted interval graphs as 
# sequence interval graphs

# documentation on NetworkX is on
# https://networkx.github.io/


import networkx as nx
import numpy as np


G=nx.Graph()

# source: http://stackoverflow.com/questions/1628949/to-find-first-n-prime-numbers-in-python
def prime(i, primes):
    for prime in primes:
        if not (i == prime or i % prime):
            return False
    primes.add(i)
    return i

def get_n_primes(n):
    primes = set([2])
    i, p = 2, 0
    while True:
        if prime(i, primes):
            p += 1
            if p == n:
                return list(primes)
        i += 1

# returns the ofset when given Pi and a list of Pml's
def geto(Pi, lisPm):
    k = int(Pi * np.prod(lisPm))
    Pm = 1
    # in description of oi: oi < Pi and oi mod Pi = 0
    for Oi in range(0, k, Pi):
        for Pml in lisPm:
            if not (Oi % Pml == 1):
                Pm = Pml
                break
        if (Oi % Pm == 1):
            return Oi

# every DIG vertex is represented as a tuple:
#           (offset, jumpsize, stepNumber)
# TODO: Implement offset and stepnumber
def convert_to_dig( G ):
    "converts a graph to a dotted interval graph"
    listOfNodes = G.nodes()
    listOfEdges = G.edges()

    N = len(listOfNodes)
    NLisPrime = get_n_primes(N)
    #NLisPrime = sieve_for_primes_to(N)
    counter=0
    
    # initialize DIG vertex
    for it in listOfNodes:
        DIG = []
        lisOfPm = []
        p0 = NLisPrime[counter]

        counter2 = 0
        # keeps track of all primes of vertices it is not connected to
        for it2 in listOfNodes:
            if counter2 < counter: break
            if (G.has_edge(it,it2) or G.has_edge(it2,it)):
                lisOfPm.append(NLisPrime[counter2])
            counter2 += 1

        print(lisOfPm)
        jumpsize = int(p0 * np.prod(lisOfPm))
        offset = geto(p0, lisOfPm)
        
        #print("vertex: " + str(it))
        #print("jumpsize:" + str(jumpsize))
        #print("offset: " + str(jumpsize))

        DIG.append((offset, jumpsize))
        counter +=1
    return DIG
