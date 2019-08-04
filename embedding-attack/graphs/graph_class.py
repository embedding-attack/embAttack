import networkx as nx
import config
import graphs.imported_scripts.snowball_sampling as snowball
import graphs.imported_scripts.metropolis_hastings_random_walk_sampling as metro
import graphs.imported_scripts.swr as swr
# import matplotlib.pyplot as plt
from typing import List, Tuple, Iterable
import functools


class Graph:
    def __init__(self, name: str, nodes: List[int], edges: List[Tuple[int, int]]):
        self._name: str = name
        self._nodes: list = list(sorted(nodes))
        self._edges: list = edges

    @staticmethod
    def init_from_list_of_edges(edges: List[Tuple[int, int]], name: str):
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        return Graph(name=name, nodes=sorted(list(nodes)), edges=list(edges))

    @staticmethod
    def init_from_networkx(g: nx.Graph):
        '''
        g = g.copy()
        # make sure node labels are int
        label_dict = dict(zip(g.nodes(), range(len(g))))
        nx.relabel_nodes(g, label_dict, copy=False)
        '''
        name = str(g)
        g = nx.convert_node_labels_to_integers(g)
        return Graph(name, list(g.nodes()), list(g.edges()))

    @staticmethod
    def init_from_gexf(path: str):
        if not path.endswith(".gexf"):
            raise ValueError("Path does not end with .gexf")

        g = nx.read_gexf(path)
        g.name = path.split("/")[-1][:-5]  # remove file suffix
        return Graph.init_from_networkx(g)

    @staticmethod
    def __get_homophily_gefx_link(graph_name: str):
        return Graph.__graph_base_dir() + f"homophily-graphs/{graph_name}.gexf"

    @staticmethod
    def __get_snowball_sampled_homophily(graph_name: str):
        return Graph.__graph_base_dir() + f"snowball_sampled_homophily/{graph_name}.edgelist"

    @staticmethod
    def __get_subsampled_homophily_gefx_link(graph_name: str):
        return Graph.__graph_base_dir() + f"subsampled_homophily/{graph_name}.gexf"

    @staticmethod
    def __get_gen_graph_link():
        return Graph.__graph_base_dir() + f"generated-graphs/"

    @staticmethod
    def __init_barabasi_graph(n, m):
        return Graph.init_from_edge_list(Graph.__get_gen_graph_link() + f"barabassi_m{m}_n{n}.edgelist")

    @staticmethod
    def __graph_base_dir():
        return f"{config.DIR_PATH}ma-graphs/"

    @staticmethod
    def init_from_edge_list(edge_list_file: str, name: str = None):
        edges: List[Tuple[int, int]] = []
        nodes = set()
        with open(edge_list_file, "r") as f:
            # assert (edge_list_file.endswith(".edgelist") or edge_list_file.endswith(".edges"))
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith("%"):
                    continue
                line = line.replace('\t', ' ')
                edge = tuple(map(int, (line.strip('\n').split(" ")[:2])))
                edges.append(edge)
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        if name is None:
            name = edge_list_file.split("/")[-1].strip(".edgelist")
        return Graph(name=name, nodes=list(nodes), edges=edges)

    '''
    Available Graphs init
    '''

    @staticmethod
    def init_karate_club_graph() -> 'Graph':
        return Graph.init_from_networkx(nx.karate_club_graph())


    @staticmethod
    def init_sampled_aps_pacs052030():
        return Graph.init_from_gexf(Graph.__get_homophily_gefx_link("sampled_APS_pacs052030"))

    @staticmethod
    def init_small_connected_subgraph_sampled_aps_pacs052030():
        return Graph.init_from_gexf(
            Graph.__get_subsampled_homophily_gefx_link("connected_subgraph_sampled_APS_pacs052030"))

    @staticmethod
    def init_DBLP_graph_moderate_homophily():
        return Graph.init_from_gexf(Graph.__get_homophily_gefx_link("DBLP_graph_moderate_homophily"))

    @staticmethod
    def init_github_mutual_follower_ntw():
        return Graph.init_from_gexf(Graph.__get_homophily_gefx_link("github_mutual_follower_ntw"))

    @staticmethod
    def init_subsample_github_mutual_follower_ntw_136():
        return Graph.init_from_gexf(
            Graph.__get_subsampled_homophily_gefx_link("subsample_github_mutual_follower_ntw_136"))

    @staticmethod
    def init_subsample_aps_pacs052030_108():
        return Graph.init_from_gexf(
            Graph.__get_subsampled_homophily_gefx_link("subsample_aps_pacs052030_108"))

    @staticmethod
    def init_subsampled_DBLP_graph_moderate_homophily_108():
        return Graph.init_from_gexf(
            Graph.__get_subsampled_homophily_gefx_link("subsampled_DBLP_graph_moderate_homophily_108"))

    @staticmethod
    def init_pok_max_cut_high_heterophily():
        return Graph.init_from_gexf(Graph.__get_homophily_gefx_link("pok_max_cut"))

    @staticmethod
    def init_local_subsample_pok_max_cut():
        return Graph.init_from_gexf(Graph.__get_subsampled_homophily_gefx_link("local_subsample_pok_max_cut"))

    @staticmethod
    def __get_facebook_link():
        return Graph.__graph_base_dir() + "facebook/"

    @staticmethod
    def init_facebook_circle(id: int):
        return Graph.init_from_edge_list(Graph.__get_facebook_link() + f"{id}.edges", name=f"facebook_circle_{id}")

    @staticmethod
    def init_facebook_circle_107():
        return Graph.init_facebook_circle(107)

    @staticmethod
    def init_list_of_homophily_graphs():
        # yield Graph.init_sampled_aps_pacs052030()
        yield Graph.init_pok_max_cut_high_heterophily()
        yield Graph.init_github_mutual_follower_ntw()
        yield Graph.init_DBLP_graph_moderate_homophily()

    @staticmethod
    def init_connected_component_sampled_aps():
        return Graph.init_from_edge_list(
            Graph.__get_snowball_sampled_homophily("connected_component_sampled_APS_pacs052030"))

    @staticmethod
    def init_pok_max_cut_snowball_sampled_2000():
        return Graph.init_from_edge_list(Graph.__get_snowball_sampled_homophily("pok_max_cut_snowball_sampled_2000"))

    @staticmethod
    def init_github_mutual_follower_ntw_snowball_sampled_2000():
        return Graph.init_from_edge_list(
            Graph.__get_snowball_sampled_homophily("github_mutual_follower_ntw_snowball_sampled_2000"),
            name="github_mutual_follower_ntw_snowball_sampled_2000")

    @staticmethod
    def init_DBLP_graph_moderate_homophily_snowball_sampled_2000():
        return Graph.init_from_edge_list(
            Graph.__get_snowball_sampled_homophily("DBLP_graph_moderate_homophily_snowball_sampled_2000"))

    @staticmethod
    def init_list_of_snowball_sampled_2000_homophily_graphs():
        yield Graph.init_connected_component_sampled_aps()
        yield Graph.init_pok_max_cut_snowball_sampled_2000()
        yield Graph.init_github_mutual_follower_ntw_snowball_sampled_2000()
        yield Graph.init_DBLP_graph_moderate_homophily_snowball_sampled_2000()
        yield Graph.init_facebook_wosn_2009_snowball_sampled_2000()

    @staticmethod
    def init_barabasi_m2_n1000():
        return Graph.__init_barabasi_graph(n=1000, m=2)

    @staticmethod
    def init_barabasi_m5_n1000():
        return Graph.__init_barabasi_graph(n=1000, m=5)

    @staticmethod
    def init_barabasi_m10_n1000():
        return Graph.__init_barabasi_graph(n=1000, m=10)

    @staticmethod
    def init_barabasi_m20_n1000():
        return Graph.__init_barabasi_graph(n=1000, m=20)

    @staticmethod
    def init_barabasi_m50_n1000():
        return Graph.__init_barabasi_graph(n=1000, m=50)

    @staticmethod
    def init_barabasi_m5_n100():
        return Graph.__init_barabasi_graph(n=100, m=5)

    @staticmethod
    def init_barabasi_m5_n500():
        return Graph.__init_barabasi_graph(n=500, m=5)

    @staticmethod
    def init_barabasi_m5_n2000():
        return Graph.__init_barabasi_graph(n=2000, m=5)

    @staticmethod
    def init_barabasi_m5_n5000():
        return Graph.__init_barabasi_graph(n=5000, m=5)

    @staticmethod
    def init_barabasi_m5_n10000():
        return Graph.__init_barabasi_graph(n=10000, m=5)

    @staticmethod
    def init_list_of_barabasi_graphs_with_different_density():
        yield Graph.init_barabasi_m2_n1000()
        yield Graph.init_barabasi_m5_n1000()
        yield Graph.init_barabasi_m10_n1000()
        yield Graph.init_barabasi_m20_n1000()
        yield Graph.init_barabasi_m50_n1000()

    @staticmethod
    def init_list_of_barabasi_graphs_with_different_size():
        yield Graph.init_barabasi_m5_n100()
        yield Graph.init_barabasi_m5_n500()
        yield Graph.init_barabasi_m5_n1000()
        yield Graph.init_barabasi_m5_n2000()
        yield Graph.init_barabasi_m5_n5000()
        # yield Graph.init_barabasi_m5_n10000()

    @staticmethod
    def init_all_but_barabasi():
        yield Graph.init_facebook_wosn_2009_snowball_sampled_2000()
        yield Graph.init_hamsterster_cc()
        # yield Graph.init_email_eu_core_cc()
        # yield Graph.init_connected_component_sampled_aps()
        yield Graph.init_DBLP_graph_moderate_homophily_snowball_sampled_2000()
        #yield Graph.init_github_mutual_follower_ntw_snowball_sampled_2000()
        #yield Graph.init_pok_max_cut_snowball_sampled_2000()

    @staticmethod
    def init_all_different_graphs():
        yield Graph.init_barabasi_m5_n1000()
        yield from Graph.init_all_but_barabasi()

    @staticmethod
    def init_all_barabasi_graphs():
        yield from Graph.init_list_of_barabasi_graphs_with_different_density()
        yield from Graph.init_list_of_barabasi_graphs_with_different_size()

    @staticmethod
    def init_list_of_all_used_graphs():
        yield from Graph.init_all_but_barabasi()
        yield from Graph.init_list_of_barabasi_graphs_with_different_density()
        yield from Graph.init_list_of_barabasi_graphs_with_different_size()

    @staticmethod
    def init_facebook_wosn_2009() -> "Graph":
        return Graph.init_from_edge_list(Graph.__graph_base_dir() + "facebook-links-wosn.edgelist",
                                         name="facebook_wosn")

    @staticmethod
    def init_hamsterster_cc() -> "Graph":
        return Graph.init_from_edge_list(Graph.__graph_base_dir() + "hamsterster_cc.edgelist",
                                         name="hamsterster_cc")

    @staticmethod
    def init_facebook_wosn_2009_snowball_sampled_2000() -> "Graph":
        return Graph.init_from_edge_list(Graph.__graph_base_dir() + "facebook_wosn_snowball_sampled_2000.edgelist",
                                         name="facebook_wosn_snowball_sampled_2000")

    @staticmethod
    def init_email_eu_core_cc() -> "Graph":
        """
            Source : https://snap.stanford.edu/data/email-Eu-core.html
        """
        return Graph.init_from_edge_list(Graph.__graph_base_dir() + "email-Eu-core_cc.edgelist",
                                         name="email_eu_core_cc")

    def edges(self):
        return self._edges

    def nodes(self):
        return self._nodes

    def name(self):
        return self._name

    def __str__(self):
        return self._name

    @functools.lru_cache(maxsize=None, typed=False)
    def neighbours(self, node: int):
        neighbours: list = []
        for edge in self._edges:
            if node in edge:
                neighbours.append(edge[0] if edge[1] == node else edge[1])
        return neighbours

    def two_hop_neighbours(self, node: int):
        neighbours: list = self.neighbours(node)
        two_hop_neighbours = set(neighbours)
        for neighbour in neighbours:
            two_hop_neighbours = two_hop_neighbours.union(set(self.neighbours(neighbour)))
        return list(two_hop_neighbours)

    def delete_node(self, removed_node: int):
        new_nodes = self.nodes().copy()
        try:
            new_nodes.remove(removed_node)
        except ValueError:
            raise ValueError("Node {} is not in the graph, hence can not be removed!".format(removed_node))

        new_edges = list(filter(lambda edge: edge[0] != removed_node and edge[1] != removed_node, self.edges()))

        return Graph(name=self._name, nodes=new_nodes, edges=new_edges)

    def add_fully_connected_node(self, node_name: int, inplace: bool = False):
        if node_name in self.nodes():
            raise ValueError("Node {} is already in the Graph. Nodes:{}".format(node_name, self.nodes()))

        new_edges = [(node_name, node) for node in self.nodes()]

        if inplace:
            self._edges.extend(new_edges)
            self._nodes.append(node_name)
        else:
            e = self.edges().copy()
            n = self.nodes().copy()
            e.extend(new_edges)
            n.append(node_name)
            return Graph(self._name, n, e)
        # self._edges.append()

    def copy(self) -> 'Graph':
        n = self.nodes().copy()
        e = self.edges().copy()
        return Graph(name=self.name(), nodes=n, edges=e)

    def to_networkx(self) -> nx.Graph:
        nx_g = nx.Graph()
        nx_g.add_nodes_from(self.nodes())
        nx_g.add_edges_from(self.edges())
        return nx_g

    def degree(self, node: int):
        return len(self.neighbours(node))

    def all_degrees(self):
        return list(map(lambda node: self.degree(node), self.nodes()))

    def get_neighbour_dict(self):
        d = dict()
        for node in self.nodes():
            d[node] = []

        for edge in self.edges():
            d[edge[0]].append(edge[1])
            d[edge[1]].append(edge[0])
        return d

    def average_neighbour_degree(self, node):
        neighbours = self.neighbours(node)
        deg_sum = 0
        for neighbour in self.neighbours(node):
            deg_sum += self.degree(neighbour)

        return deg_sum / len(neighbours)

    def distance(self, node1, node2):
        """
        Waring: Very Inefficient
        :param node1: 
        :param node2: 
        :return: distance between node1 and node2
        """
        try:
            dist = nx.shortest_path_length(self.to_networkx(), source=node1, target=node2)
        except:
            dist = 9999
        return dist

    def snowball_sampling(self, center_node: int = None, maxsize: int = 50, name=None):
        g_x = self.to_networkx()

        if center_node is None:
            center_node = snowball.randomseed(g_x)
        else:
            assert (center_node in self.nodes())

        subgraph = snowball.snowballsampling(g=g_x, seed=center_node, maxsize=maxsize)
        sub_x: nx.Graph = g_x.subgraph(subgraph)
        if name is not None:
            sub_x.name = name
        return Graph.init_from_networkx(sub_x)

    def sampling_metropolis_hastings_random_walk(self, subgraph_size: int, seed_node: int = None, name=None):
        '''
        sampling graph using unform metropolish hashing random walk
        :param subgraph_size: size of target subgraph
        :param seed_node: optional seed node for sampling
        :param name: name of the target graph (default self.name + '_MHRW_{subgraph_size}'
        :return: subgraph
        '''
        g_x = self.to_networkx()

        nx_subgraph = metro.metropolis_hastings_random_walk_uniform(graph=g_x, subgraph_size=subgraph_size,
                                                                    seed_node=seed_node)

        if name is not None:
            nx_subgraph.name = name
        else:
            nx_subgraph.name = str(self) + f"_MHRW_{subgraph_size}"

        return Graph.init_from_networkx(nx_subgraph)

    def random_walk_induced_graph_sampling(self, subgraph_size: int, seed_node: int = None, name=None):
        '''
       sampling graph using unform metropolish hashing random walk
       :param subgraph_size: size of target subgraph
       :param seed_node: optional seed node for sampling
       :param name: name of the target graph (default self.name + '_MHRW_{subgraph_size}'
       :return: subgraph
       '''
        g_x = self.to_networkx()
        nx_subgraph = swr.SRW_RWF_ISRW().random_walk_induced_graph_sampling(complete_graph=g_x,
                                                                            nodes_to_sample=subgraph_size)

        if name is not None:
            nx_subgraph.name = name
        else:
            nx_subgraph.name = str(self) + f"_ISRW_{subgraph_size}"

        return Graph.init_from_networkx(nx_subgraph)

    def betweenness_centrality(self):
        return nx.betweenness_centrality(self.to_networkx())

    def remove_self_loops(self):
        self._edges = list(filter(lambda e: e[0] != e[1], self.edges()))

    def is_connected(self):
        return nx.is_connected(self.to_networkx())

    @functools.lru_cache(maxsize=32)
    def splits_graph(self, node: int):
        """
        tests if the graph is split by removing node "node"
        :param node: the node that might split the graph
        :return: bool if node splits graph. True if it splits the graph
        :return: bool if node splits graph. True if it splits the graph
        """
        gnx = self.to_networkx()
        gnx.remove_node(n=node)

        return not nx.is_connected(gnx)

    def add_node(self, neighbours: Iterable[int]) -> "Graph":
        new_node_name = len(self.nodes())
        new_edges = list(map(lambda n: (n, new_node_name), neighbours))

        return Graph(name=self.name() + f'_added_node_neighbours_{neighbours}', nodes=self.nodes() + [new_node_name],
                     edges=self.edges() + new_edges)

    def density(self):
        return nx.density(self.to_networkx())

    def triangle_count(self):
        return sum(nx.triangles(self.to_networkx()).values()) / 3