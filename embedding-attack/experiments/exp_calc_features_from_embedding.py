#!/usr/bin/env python

from distance_matrices import calculate_distance_matrix as cd
from features import create_features as cf, feature_type as ft
import memory_access as sl
import utils
import graphs.graph_class as gc
import multiprocessing
import functools
import config
import numpy as np
import features.diff_type as dt
import pandas as pd
import experiments.exp_calc_features_from_similarity_diff as exp_sim_diff
import experiments.exp_utils as exp_utils

def __calc_dm(graph: gc.Graph, removed_nodes: [int], save_info: sl.MemoryAccess, i: int) -> (int, pd.DataFrame):
    if save_info.has_distance_matrix(removed_nodes=removed_nodes, iteration=i):
        # print("Distance matrix for removed nodes", removed_nodes, "and iteration", i, "is already trained")
        return i, save_info.load_distance_matrix(removed_nodes=removed_nodes, iteration=i)
    else:
        # print(f'Calculate distance matrix for removed nodes {removed_nodes} iteration {i}')
        # model = embedding_function(graph=graph, save_info=save_info, removed_nodes=removed_nodes, iteration=i)
        # thows error if embedding does not exist

        model = save_info.load_embedding(removed_nodes=removed_nodes, iteration=i)
        dm = cd.calc_distances(model=model, graph=graph, save_info=save_info, removed_nodes=removed_nodes, iteration=i,
                               save=False)

        save_info.save_distance_matrix(removed_nodes=removed_nodes, iteration=i, dm=dm)

        return i, dm


def calc_avg_distance_matrix(graph: gc.Graph,
                             removed_nodes: [int],
                             save_info: sl.MemoryAccess):
    if save_info.has_avg_distance_matrix(removed_nodes=removed_nodes):
        save_info.delete_distance_matrices(removed_nodes=removed_nodes)
        return save_info.load_avg_distance_matrix(removed_nodes)

    used_embeddings = range(save_info.get_num_iterations())

    avg_dm = pd.DataFrame(0.0, index=graph.nodes(), columns=graph.nodes())

    dm_calc_func = functools.partial(__calc_dm, graph, removed_nodes, save_info)

    for iter in used_embeddings:
        res = dm_calc_func(iter)
        i, dm = res
        utils.assure_same_labels([avg_dm, dm],
                                 "Format of distance matrix iteration {} \
                                 for removed nodes  {} is not correct".format(i, removed_nodes))
        avg_dm += dm

    avg_dm = avg_dm.div(len(used_embeddings))
    # save avg distance matrix
    save_info.save_avg_distance_matrix(removed_nodes, avg_dm)
    # delete dms for memory space
    save_info.delete_distance_matrices(removed_nodes=removed_nodes)
    return avg_dm


def __compute_training_features_for_one_node(dm_original: pd.DataFrame,
                                             node_to_predict: int,
                                             save_info: sl.MemoryAccess, graph: gc.Graph,
                                             num_of_bins: int, feature_type: ft.FeatureType,
                                             nodes_to_train_on: [int]) -> None:
    """
    :param dm_original: distance matrix of the original graph
    :param node_to_predict: node that is removed from the graph and should be predicted
    :param save_info: data access object
    :param graph: graph the embedding is trained on
    :param num_of_bins: number of bins that should be used to generate training features
    :param feature_type: type of the feature vector that is used
    :param nodes_to_train_on: a list of nodes that are removed from the graph after removing
            node_to_predict to generate training data
    """

    # --- compute test features for node_to_predict ---
    # remove node_to_predict from the graph
    graph_reduced = graph.delete_node(node_to_predict)
    dm_reduced = calc_avg_distance_matrix(graph=graph_reduced,
                                          removed_nodes=[node_to_predict],
                                          save_info=save_info)

    # test if training data is already avialable
    if save_info.has_training_data([node_to_predict], feature_type=feature_type, num_of_bins=num_of_bins):
        # print("Training Feature for removed nodes ", [node_to_predict], " and feature type ",
        #     "diff_bins_num:" + str(num_of_bins) + "and_norm_dim", "is already trained")
        pass
    else:
        # print(f"Compute test features for node {node_to_predict}")
        diff = cf.create_difference_matrix(dm_original, dm_reduced, removed_nodes=[node_to_predict],
                                           save_info=save_info)

        # compute training data
        # cf.create_feature_from_diff_bins_with_dim(diff=diff, removed_nodes=[node_to_predict], original_graph=graph,
        #                                          num_of_bins=num_of_bins, save_info=save_info)
        cf.create_features(diff=diff, removed_nodes=[node_to_predict], original_graph=graph, num_of_bins=num_of_bins,
                           feature_type=feature_type, save_info=save_info)

        del diff  # free RAM
        # save_info.remove_diff_matrix(removed_nodes=[node_to_predict])  # free memory

    # --- compute training features for nodes_to_train_on ---
    # print(f"Create training features for removed node {node_to_predict} by by removing ", nodes_to_train_on)
    for node in nodes_to_train_on:

        # check if features already exists
        if save_info.has_training_data(removed_nodes=[node_to_predict, node],
                                       feature_type=feature_type, num_of_bins=num_of_bins):
            # print("Training Feature for removed nodes ", [node_to_predict, node], " and feature type ",
            #     "diff_bins_num:" + str(num_of_bins) + "and_norm_dim", "is already trained")
            pass
        else:
            graph_reduced_2 = graph_reduced.delete_node(node)
            dm_reduced_2 = calc_avg_distance_matrix(graph=graph_reduced_2,
                                                    removed_nodes=[node_to_predict, node],
                                                    save_info=save_info)
            print("odm", type(dm_reduced), "rdm", type(dm_reduced_2))
            diff_reduced = cf.create_difference_matrix(dm_reduced, dm_reduced_2, removed_nodes=[node_to_predict, node],
                                                       save_info=save_info)


            print("rdiff", type(diff_reduced), "odm", type(dm_reduced), "rdm", type(dm_reduced_2))
            del dm_reduced_2
            # compute training data

            cf.create_features(diff=diff_reduced, removed_nodes=[node_to_predict, node],
                               original_graph=graph_reduced,
                               num_of_bins=num_of_bins, save_info=save_info, feature_type=feature_type)





def __compute_training_features_for_one_node_pool(dm_original: pd.DataFrame,
                                                  save_info: sl.MemoryAccess, graph: gc.Graph,
                                                  num_of_bins: int, feature_type: ft.FeatureType,
                                                  nodes_to_train_on: {}, node_to_predict: int):
    __compute_training_features_for_one_node(dm_original=dm_original, node_to_predict=node_to_predict,
                                             save_info=save_info, graph=graph,
                                             num_of_bins=num_of_bins, feature_type=feature_type,
                                             nodes_to_train_on=nodes_to_train_on[node_to_predict])


def compute_training_features(save_info: sl.MemoryAccess, graph: gc.Graph, list_nodes_to_predict: [int],
                              nodes_to_train_on: {},
                              num_of_bins: int, feature_type: ft.FeatureType = None,num_eval_iterations:int = None):
    """
    :param save_info: memory access obj
    :param graph: graph the embedding is trained on (used to access nodes lists)
    :param num_of_bins: number of bins the feature vector should use
    :param feature_type: type of the features to compute
    :param list_nodes_to_predict: nodes that are used as test_cases.
            If None nodes are determined by available files in the file system
    :param nodes_to_train_on: nodes that are used for training in each test case. Dict from the node_to_predict to [int]
            containin the training nodes for that tested node.
            If None
    """

    print(f"Compute training features on diff type {save_info.get_diff_type()} and graph {str(graph)} "
          f"on nodes {list_nodes_to_predict} "
          f" graph  embedding {str(save_info.embedding_type)}")


    if save_info.get_diff_type().has_one_init_graph():
        if num_eval_iterations is None:
            iteration_values = list(range(save_info.get_num_iterations()))
        else:
            iteration_values = list(range(num_eval_iterations))
    else:
        iteration_values = [-1]

    if feature_type is None:
        feature_type = ft.FeatureType.DIFF_BIN_WITH_DIM

    for diff_iter in iteration_values:
        if diff_iter != -1:
            save_info.get_diff_type().set_iter(diff_iter)



        p_nodes = list_nodes_to_predict
        t_nodes = nodes_to_train_on

        if save_info.get_diff_type() in [dt.DiffType.MOST_SIMILAR_EMBS_DIFF,
                                         dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ALL_EMBS,
                                         dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT,
                                         dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE]:
            exp_sim_diff.compute_training_features_from_similarity_diff(save_info=save_info, graph=graph,
                                                                        num_of_bins=num_of_bins,
                                                                        feature_type=feature_type,
                                                                        p_nodes=p_nodes,
                                                                        t_nodes=t_nodes)
        else:

            num_features = len(p_nodes)

            p_nodes, t_nodes = exp_utils.filter_by_already_trained_nodes(
                p_node_list=p_nodes, t_node_dict=t_nodes, graph=graph,
                save_info=save_info, feature_type=feature_type, num_bins=num_of_bins)

            if len(p_nodes) > 0:
                # compute distance matrix of the original graph
                if save_info.get_diff_type() == dt.DiffType.DIFFERENCE_ONE_INIT:
                    emb_number = save_info.get_diff_type().get_iter()
                    if emb_number == -1 or emb_number is None:
                        raise ValueError(f"The selected Difference Type requires an iteration number. "
                                         f"E.g. dt.DiffType.DIFFERENCE_ONE_INIT.set_iter(0).")

                    _, dm_original = __calc_dm(graph=graph, removed_nodes=[], save_info=save_info, i=emb_number)
                elif save_info.get_diff_type() == dt.DiffType.DIFFERENCE:
                    dm_original = calc_avg_distance_matrix(graph=graph, removed_nodes=[], save_info=save_info)
                else:
                    raise ValueError(f"Invalid Difference Type: {save_info.get_diff_type()}")

                func_p = functools.partial(__compute_training_features_for_one_node_pool, dm_original,
                                           save_info, graph,
                                           num_of_bins, feature_type,
                                           t_nodes)


                with multiprocessing.Pool(min(config.NUM_CORES, len(p_nodes))) as pool:
                    for res in pool.imap(func_p, p_nodes):
                        pass
                '''
                for i in p_nodes:
                    func_p(i)
                '''
            else:
                if num_features == 0:
                    raise ValueError("no embeddings found to create training features for")
                else:
                    print(f"All features are already trained. Number of training features {num_features}")


if __name__ == '__main__':
    import embeddings.node2vec_felix as n2v

    graph = gc.Graph.init_karate_club_graph()
    emb = n2v.Node2VecF()

    save_info = sl.MemoryAccess(graph=str(graph), embedding_type="Node2Vec", num_iterations=30,
                                diff_type=dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE.set_iter(5))

    compute_training_features(save_info=save_info, graph=graph, num_of_bins=10,
                              list_nodes_to_predict=[0], nodes_to_train_on={0: [1, 2]})
