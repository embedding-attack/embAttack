import networkx as nx
import queue as que
import copy
import numpy.random as random

def randomseed(g):
    """ this function recturns a single node from g, it's chosen with uniform probability """
    ux = random.randint(0,g.number_of_nodes(),1)
    return ux[0]

def snowballsampling(g, seed, maxsize=50):
    """ this function returns a set of nodes equal to maxsize from g that are collected from around seed node via
        snownball sampling """
    if g.number_of_nodes() < maxsize:
        raise ValueError(f"Graph is smaller than the number of maximal nodes. Graph size {g.number_of_nodes}")

    q = que.Queue()
    q.put(seed)
    subgraph = {seed}
    while not q.empty():
        neighbours = list(g.neighbors(q.get()))
        random.shuffle(neighbours)
        for node in neighbours:
            if len(subgraph) < maxsize:
                q.put(node)
                subgraph.add(node)
            else :
                return subgraph
            pass
        pass
    return subgraph

def surroundings(g, subgraph):
    """ this function returns the surrounding subgraph of input subgraph argument """ 
    surdngs = copy.copy(subgraph)
    for node in subgraph:
        for i in g.neighbors(node):
            if i not in surdngs:
                surdngs.append(i)
                pass
            pass
        pass
    return surdngs