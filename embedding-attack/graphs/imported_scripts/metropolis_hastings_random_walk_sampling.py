import networkx as nx
import numpy as np
import tqdm

def metropolis_hastings_random_walk_uniform(graph:nx.Graph, subgraph_size:int, seed_node:int=None):
    '''
    Implementaion is based on pseudo code in https://www.researchgate.net/profile/Minas_Gjoka/publication/224137028_Walking_in_Facebook_A_Case_Study_of_Unbiased_Sampling_of_OSNs/links/09e41507daffd4ac8d000000/Walking-in-Facebook-A-Case-Study-of-Unbiased-Sampling-of-OSNs.pdf
    :param graph: graph that should be sampled
    :param seed_node: inital seed node
    :param subgraph_size: size of the subgraph
    :return:
    '''

    if seed_node is None:
        seed_deg = -1
        while seed_deg <= 0:
            seed_node = np.random.randint(low=0, high=graph.number_of_nodes())
            seed_deg = graph.degree(seed_node)

    # set that contains all nodes that shoudl be in subgraph
    sampled_nodes = set([seed_node])

    # current node in random walk
    current_node = seed_node
    current_node_degree = graph.degree(current_node)

    with tqdm.tqdm(total=subgraph_size) as pbar:
        pbar.update(1)
        while len(sampled_nodes) < subgraph_size:
            neighbours = nx.neighbors(graph,current_node)

            #filter by already added nodes
            #neighbours = list(filter(lambda n: n in sampled_nodes,neighbours))
            #assert(len(neighbours) > 0)

            # chosen candidate
            candidate = np.random.choice(a=list(neighbours),size=1)[0]
            candidate_degree = graph.degree(candidate)

            #generate random number that determines if node is selected
            p = np.random.rand()
            a = (current_node_degree/candidate_degree)
            #if node is selected add it to sample
            if p <= (current_node_degree/candidate_degree):
                sampled_nodes.add(candidate)
                current_node = candidate
                current_node_degree = candidate_degree
                pbar.update(1)

    assert(len(sampled_nodes) == subgraph_size)

    return nx.subgraph(graph,sampled_nodes)