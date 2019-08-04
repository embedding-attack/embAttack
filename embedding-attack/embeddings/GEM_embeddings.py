import sys
import config

sys.path.insert(0, config.GEM_PATH + "GEM")
import gem.embedding.static_graph_embedding as abs_emb

import embeddings.embedding
import abc
import networkx as nx
import graphs.graph_class as gc
import memory_access as sl
import pandas as pd
import numpy as np

import gem.embedding.static_graph_embedding as abs_emb
import gem.embedding.lle as lle
import gem.embedding.lap as lap
import gem.embedding.gf as gf
import gem.embedding.hope  as hope
import gem.embedding.sdne  as sdne
import typing


class GEM_embedding(embeddings.embedding.Embedding):  # metaclass=abc.ABCMeta

    def __init__(self, embedding: abs_emb,is_static:bool = False):
        self.__gem_embedding: abs_emb = embedding
        self.__is_static = is_static
    def __str__(self):
        return f"{self.__gem_embedding.get_method_summary()}"

    def short_name(self):
        return f"{self.__gem_embedding.get_method_name()}"
    def train_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int],
                        num_of_embeddings: int):
        super().train_embedding(graph=graph, save_info=save_info, removed_nodes=removed_nodes,
                                num_of_embeddings=num_of_embeddings)

        nx_g = graph.to_networkx()
        nx_g.to_directed()

        np.testing.assert_array_equal(nx_g.nodes(), graph.nodes())
        nx_g = nx.convert_node_labels_to_integers(nx_g)

        for iter in range(num_of_embeddings):
            if save_info.has_embedding(removed_nodes=removed_nodes,iteration=iter):
                continue

            Y, t = self.__gem_embedding.learn_embedding(graph=nx_g, is_weighted=False, no_python=True)

            emb = pd.DataFrame(Y, index=graph.nodes())

            save_info.save_embedding(removed_nodes=removed_nodes, iteration=iter, embedding=emb)



    def load_embedding(self, graph: gc.Graph, removed_nodes: [int], save_info, iteration: int,
                       load_neg_results: bool = False):
        pass

    def continue_train_embedding(self, graph: gc.Graph,
                                 save_info, removed_nodes: [int],
                                 num_of_embeddings: int, model, emb_description: str = None,
                                 graph_description: str = None):
        pass

    @staticmethod
    def init_local_linear_embedding(dim: int = 128):
        """
        local linear embedding. Does not have variance in embeddings, hence one iteration is sufficiant
        :param dim:
        :return:
        """
        return GEM_embedding(lle.LocallyLinearEmbedding(d=dim),is_static=True)

    @staticmethod
    def init_graph_factorisation(dim: int = 128, max_iter: int = 1000, eta: float = 1 * 10 ** -4, regu: float = 1.0):
        """
        no variance im embedding
        :param dim:
        :param max_iter:
        :param eta:
        :param regu:
        :return:
        """
        return GEM_embedding(gf.GraphFactorization(d=dim, max_iter=max_iter, eta=eta, regu=regu),is_static=True)

    @staticmethod
    def init_hope(dim: int = 128, beta: float = 0.01):
        """
        very small variance
        :param dim:
        :param beta:
        :return:
        """
        return GEM_embedding(hope.HOPE(d=dim, beta=beta))

    @staticmethod
    def init_sdne(dim: int = 128, beta: int = 5, alpha: float = 1e-5, nu1: float = 1e-6, nu2: float = 1e-6, K: int = 3,
                  n_units: typing.List[int] = [500, 300, ], rho: float = 0.3, n_iter: int = 30, xeta: float = 0.001,
                  n_batch: int = 500):
        """
        large variance
        :param dim:
        :param beta:
        :param alpha:
        :param nu1:
        :param nu2:
        :param K:
        :param n_units:
        :param rho:
        :param n_iter:
        :param xeta:
        :param n_batch:
        :return:
        """
        return GEM_embedding(
            sdne.SDNE(d=dim, beta=beta, alpha=alpha, nu1=nu1, nu2=nu2, K=K, n_units=n_units, rho=rho, n_iter=n_iter,
                      xeta=xeta, n_batch=n_batch))

    @staticmethod
    def init_list_of_gem_embeddings():
        yield GEM_embedding.init_hope()
        yield GEM_embedding.init_sdne()
        yield GEM_embedding.init_local_linear_embedding()
        yield GEM_embedding.init_graph_factorisation()


    def is_static(self):
        return self.__is_static