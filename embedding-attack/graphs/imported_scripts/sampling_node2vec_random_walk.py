import tqdm
import networkx as nx
import numpy as np
import math
from typing import List
import config

import memory_access as sl
import graphs.graph_class as gc
import graphs.graph_class_plot as gcp
import embeddings.node2vec_c_path_gensim_emb as n2v


def find_index_of_closest_node(graph: nx.Graph, target_node: int, candidates: List[int]):
    distances = list(map(lambda c: nx.shortest_path_length(graph, source=target_node, target=c), candidates))
    closest_c_index = np.argmin(distances)[0]

    return closest_c_index


def sampling_rw_n2v(graph: nx.Graph, subgraph_size: int, p: int, q: int, seed_node: int = None):
    if seed_node is None:
        seed_deg = -1
        while seed_deg <= 0:
            seed_node = np.random.randint(low=0, high=graph.number_of_nodes())
            seed_deg = graph.degree(seed_node)

    # set that contains all nodes that shoudl be in subgraph
    sampled_nodes = set([seed_node])

    # current node in random walk

    current_node = seed_node

    prev_node = None

    with tqdm.tqdm(total=subgraph_size) as pbar:
        pbar.update(1)

        while len(sampled_nodes) < subgraph_size:
            neighbours = list(nx.neighbors(graph, current_node))

            # filter by already added nodes
            # neighbours = list(filter(lambda n: n in sampled_nodes,neighbours))
            # assert(len(neighbours) > 0)

            # pronabilities for neighbours
            probs = [1 / q] * len(neighbours)

            # find clostest neighbour
            if prev_node is not None:
                closest_n_index = find_index_of_closest_node(graph=graph, target_node=prev_node, candidates=neighbours)
                probs[closest_n_index] = 1

            probs += [1 / p]
            probs = np.array(probs)/sum(probs)
            # chosen candidate
            next_node = np.random.choice(a=neighbours + [prev_node], size=1, p=probs)[0]

            sampled_nodes.add(next_node)
            prev_node = current_node
            current_node = next_node
            pbar.update(1)

    assert (len(sampled_nodes) == subgraph_size)

    return nx.subgraph(graph, sampled_nodes)


if __name__ == '__main__':
    graph = gc.Graph.init_karate_club_graph()
    sub = sampling_rw_n2v(graph=graph.to_networkx(), subgraph_size=10, p=4, q=2)
    gcp.GraphPlot.draw_graph(sub)
