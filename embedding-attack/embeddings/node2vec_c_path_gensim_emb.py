import subprocess
import pandas as pd
import os
from graphs.graph_class import Graph
import embeddings.embedding
import config
import re
import gensim
import memory_access as sl
import os.path



class Node2VecPathSnapEmbGensim(embeddings.embedding.Embedding):

    def __init__(self, dim: int = 128, epochs: object = 5, window_size: int = 10, walk_length: int = 80,
                 num_of_walks_per_node: int = 10, alpha: float = 0.025):
        self.dim: int = dim
        self.epochs: int = epochs
        self.window_size:int = window_size
        self.walk_length: int = walk_length
        self.num_of_walks_per_node: int = num_of_walks_per_node
        self.alpha:float = alpha

    def __str__(self):
        if self.alpha == 0.025:
            return 'Node2Vec_path_c_emb_gensim-dim={}_epochs={}-windowSize={}-walkLength={}-walksPerNode={}_p=1_q=1' \
                .format(self.dim,
                        self.epochs,
                        self.window_size,
                        self.walk_length,
                        self.num_of_walks_per_node)
        else:
            return 'Node2Vec_path_c_emb_gensim-dim={}_epochs={}-windowSize={}-walkLength={}\
                -walksPerNode={}_p=1_q=1_alpha_{}' \
                .format(self.dim,
                        self.epochs,
                        self.window_size,
                        self.walk_length,
                        self.num_of_walks_per_node,
                        self.alpha)

    def short_name(self):
        return "Node2Vec"

    def train_embedding(self, graph: Graph, save_info: sl.MemoryAccess, removed_nodes: [int], num_of_embeddings: int,
                        check_for_existing: bool = True):
        super().train_embedding(graph=graph, save_info=save_info, removed_nodes=removed_nodes,
                                num_of_embeddings=num_of_embeddings)

        edge_list_path = save_info.access_edge_list(graph=graph, removed_nodes=removed_nodes)

        for iteration in range(num_of_embeddings):
            train_node2vec_embedding(edge_list_path=edge_list_path, save_info=save_info, removed_nodes=removed_nodes,
                                     iteration=iteration,
                                     epochs=self.epochs, dim=self.dim, window_size=self.window_size,
                                     walk_length=self.walk_length,
                                     alpha=self.alpha,
                                     num_of_walks_per_node=self.num_of_walks_per_node, return_embedding=False,
                                     graph=graph, check_for_existing=check_for_existing)

    def load_embedding(self, graph: Graph, removed_nodes: [int], save_info: sl.MemoryAccess, iteration: int,
                       load_neg_results: bool = False):
        target = save_info.get_embedding_name(removed_nodes=removed_nodes, iteration=iteration)
        target_name = os.path.abspath(target + ".emb")
        target_name_neg = os.path.abspath(target + "_neg.emb")
        if load_neg_results:
            return load_results(target_name=target_name, node_names=graph.nodes()), load_results(
                target_name=target_name_neg, node_names=graph.nodes())
        else:
            return load_results(target_name=target_name, node_names=graph.nodes())

    def continue_train_embedding(self, model, graph: Graph, emb_description: str,
                                 graph_description: str, save_info: sl.MemoryAccess, removed_nodes: [int],
                                 num_of_embeddings: int):
        raise NotImplementedError()

    @staticmethod
    def _save_embedding(file_name: str, emb: pd.DataFrame):
        header = " ".join(map(str, emb.shape)) + "\n"
        emb_string = header + re.sub("\s\s+", " ", emb.to_string(header=False, index=True))
        with open(file_name, "w+") as file:
            file.write(emb_string)

    def is_static(self):
        return False
def load_results(target_name: str, node_names):
    """
    Returns the embeddings. Some nodes may have a feature vector of 0, if they are not connected to any other node.
    This should be sufficient for the purpose since the differences to a node that was connected but is not anymore
    should be large.

    :param target_name: path to the file that contains the data
    :param node_names: names of all nodes in the embedding
    :return: embedding:pd.Dataframe


    """
    with open(target_name, "r") as file:
        # init
        feature_length: int = int(file.readline().strip().split(" ")[1])  # line that give file info
        embedding = pd.DataFrame(0, index=node_names, columns=list(range(feature_length)))

        # fill features
        for line in file:
            # feature_vector constists of feature_vector[0]= node_name and feature_vector[1:]= features
            feature_vector = list(map(float, line.strip().split(" ")))

            series = pd.Series(feature_vector[1:])

            assert (feature_vector[0] in embedding.index)
            embedding.loc[feature_vector[0]] = series

    return embedding


def train_node2vec_embedding(edge_list_path: str,
                             graph: Graph,
                             save_info: sl.MemoryAccess,
                             removed_nodes: [int],
                             iteration: int,
                             epochs: int,
                             dim: int,
                             walk_length: int,
                             num_of_walks_per_node: int,
                             window_size: int,
                             alpha: float,
                             return_embedding: bool = False, check_for_existing: bool = True):
    target = save_info.get_embedding_name(removed_nodes=removed_nodes, iteration=iteration)

    if check_for_existing and os.path.exists(target + ".emb"):
        #print('Embedding for removed nodes {} and iteration {} already exists'.format(removed_nodes, iteration))
        if return_embedding:
            return save_info.load_embedding(removed_nodes=removed_nodes, iteration=iteration)
    else:
        target_path = os.path.abspath(target + "_path.emb")

        # create walks

        # execute path training
        wd = os.getcwd()
        os.chdir(config.NODE2VEC_SNAP_DIR)

        subprocess.call('./node2vec \
            -i:"' + edge_list_path + '" \
            -o:"' + target_path + '" \
            -e:' + str(epochs) +
                        " -d:" + str(dim) +
                        " -l:" + str(walk_length) +
                        " -r:" + str(num_of_walks_per_node) +
                        " -k:" + str(window_size) +
                        " -ow", shell=True)  # output random walks only
        os.chdir(wd)

        # end create paths

        class Walks:
            def __init__(self, file):
                self.file = file

            def __iter__(self):
                with open(target_path, "r") as f:
                    for line in f:
                        line = line.strip("\n").split(" ")
                        # assert (all(list(map(lambda node: node in graph.nodes(), list(map(int, line))))))
                        yield line

        walks = Walks(target_path)

        # train word2vec
        emb_result = gensim.models.Word2Vec(walks, size=dim, iter=epochs, window=window_size, min_count=1, sg=1,
                                            workers=config.NUM_CORES, alpha=alpha)

        os.remove(target_path)

        save_info.save_embedding(removed_nodes, iteration, emb_result)

        if return_embedding:
            return emb_result



def continue_train_node2vec_embedding(edge_list_path: str,
                                      base_embedding_pos_path: str,
                                      base_embedding_neg_path: str,
                                      save_info: sl.MemoryAccess,
                                      removed_nodes: [int],
                                      iteration: int,
                                      epochs: int,
                                      dim: int,
                                      walk_length: int,
                                      num_of_walks_per_node: int,
                                      window_size: int,
                                      emb_description: str = None,
                                      return_embedding: bool = False,
                                      graph: Graph = None):
    pass


if __name__ == "__main__":
    os.chdir("..")

    removed_node = 6
    num_of_embeddings = 1

    emb = Node2VecPathSnapEmbGensim()
    base_graph = Graph.init_karate_club_graph()

    graph = base_graph.delete_node(removed_node)
    save_info = sl.MemoryAccess(graph=str(graph), embedding_type="Node2Vec_Test", num_iterations=num_of_embeddings)

    emb.train_embedding(graph=graph, save_info=save_info, removed_nodes=[removed_node],
                        num_of_embeddings=num_of_embeddings)
    # res = emb.load_embedding(graph=graph,save_info=save_info,iteration=0)
    # print(res)
