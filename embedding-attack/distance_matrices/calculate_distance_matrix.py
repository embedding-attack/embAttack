import pandas as pd
import memory_access as sl
import utils
from scipy.spatial import distance
import scipy
import graphs.graph_class as gc
import gensim
import typing


def calculate_average_distance_matrix(dms: [pd.DataFrame], removed_nodes: [int], save_info: sl.MemoryAccess,
                                      save: bool = True, check_for_exising: bool = True) -> pd.DataFrame:
    if check_for_exising and save_info.has_avg_distance_matrix(removed_nodes):
        return save_info.load_avg_distance_matrix(removed_nodes)

    # all column and row labels are equal  for each dm
    utils.assure_same_labels(dms)

    labels = dms[0].columns.values.tolist()
    num_dms = len(dms)
    avg_dm = pd.DataFrame(0.0, labels, labels)

    for i in range(len(labels)):
        for j in range(i):
            sum_of_distances = 0

            label1 = labels[i]
            label2 = labels[j]

            for dm in dms:
                sum_of_distances += dm.at[label1, label2]
            avg_dm.at[label1, label2] = float(sum_of_distances) / num_dms

    if save:
        save_info.save_avg_distance_matrix(removed_nodes, avg_dm)

    return avg_dm


"""
def __calc_distances_based_on_df(embedding: pd.DataFrame):
    assert (sorted(embedding.index) == list(embedding.index))

    index = embedding.index
    cos = pd.DataFrame(scipy.spatial.distance.cdist(embedding, embedding, metric="cosine"), index=index,
                       columns=index)
    cos.sort_index(axis=0, inplace=True)
    cos.sort_index(axis=1, inplace=True)
    return cos
"""


def calc_distances_based_on_gensim_fast(model: typing.Union[gensim.models.keyedvectors.KeyedVectors, pd.DataFrame]):
    if type(model) == gensim.models.keyedvectors.KeyedVectors:
        embedding = model.wv.vectors
        index = list(map(int, list(model.wv.vocab.keys())))
    elif type(model) == pd.DataFrame:
        embedding = model
        index = embedding.index
    else:
        raise ValueError(f"model type is not supported! Type {type(model)} "
                         f"should be gensim.models.keyedvectors.KeyedVectors or pd.Dataframe.")

    cos = pd.DataFrame(scipy.spatial.distance.cdist(embedding, embedding, metric="cosine"), index=index,
                       columns=index)
    cos.sort_index(axis=0, inplace=True)
    cos.sort_index(axis=1, inplace=True)
    assert (cos.notna().any().any())
    # cos.fillna(0, inplace=True)

    return cos


def calc_distances(model, save_info: sl.MemoryAccess, removed_nodes: [int], iteration: int, graph: gc.Graph = None,
                   save: bool = True, check_for_existing: bool = True):
    if check_for_existing and save_info.has_distance_matrix(removed_nodes, iteration):
        return save_info.load_distance_matrix(removed_nodes, iteration)

    dm = calc_distances_based_on_gensim_fast(model=model)

    """
    if type(model) == pd.DataFrame:
        dm = __calc_distances_based_on_df(embedding=model)
    else:
        dm = calc_distances_based_on_gensim_fast(model=model)
        # dm = __calc_distances_based_on_gensim(model=model, node_names=node_names)
    """

    if save:
        save_info.save_distance_matrix(removed_nodes, iteration, dm)

    return dm


if __name__ == '__main__':
    import embeddings.transE as em_transe
    from embeddings import node2vec_gensim as em_node2vec
    import networkx as nx

    graph = nx.karate_club_graph()

    embedding_function = em_transe.train_transe_embedding
    save_info = sl.MemoryAccess(graph=str(graph), embedding_type="TransE", num_iterations=30)

    embedding_transe = em_transe.train_transe_embedding_from_networkx(nx.karate_club_graph(), save_info=save_info,
                                                                      removed_nodes=[],
                                                                      iteration=0)
    dm_transe = calc_distances(model=embedding_transe, graph=graph, save_info=save_info, removed_nodes=[], iteration=0,
                               save=False)

    embedding_node2vec = em_node2vec.train_node2vec_embedding(graph=graph, save_info=save_info, removed_nodes=[],
                                                              iteration=0, save=False)
    dm_node2vec = calc_distances(model=embedding_node2vec, graph=graph, save_info=save_info, removed_nodes=[],
                                 iteration=0, save=False)

    print("dm transe", dm_transe)
    print("dm node2vec", dm_node2vec)
    print("differences", dm_transe - dm_node2vec)
