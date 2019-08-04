import abc
import graphs.graph_class as gc
#from save_load import *
#import save_load as sl


class Embedding(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def train_embedding(self, graph: gc.Graph, save_info, removed_nodes: [int], num_of_embeddings: int):
        assert (all(node not in graph.nodes() for node in removed_nodes))

    @abc.abstractmethod
    def load_embedding(self, graph: gc.Graph, removed_nodes: [int], save_info, iteration: int,
                       load_neg_results: bool = False):
        pass


    @abc.abstractmethod
    def continue_train_embedding(self, graph: gc.Graph,
                                 save_info, removed_nodes: [int],
                                 num_of_embeddings: int, model, emb_description: str = None,
                                 graph_description: str = None):
        pass

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

    @abc.abstractmethod
    def is_static(self)->bool:
        pass