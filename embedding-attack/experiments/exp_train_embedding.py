import memory_access as sl
from graphs import graph_class as gc
from embeddings.embedding import Embedding
from experiments import exp_calc_features_from_embedding as cf
from classifier import train_sklearn_predictions as te
import utils
from features import feature_type as ft
import config
import logging
import os


def train_embedding_per_graph(graph: gc.Graph, embedding: Embedding, save_info: sl.MemoryAccess,
                              num_of_embeddings: int = 30, num_of_test_evaluations_per_degree_level: int = 5,
                              num_of_training_graphs: int = 10, num_of_bins_for_tf: [int] = None,
                              run_experiments_on_embedding: bool = True,
                              feature_type: ft.FeatureType = ft.FeatureType.DIFF_BIN_WITH_DIM):
    assert (num_of_embeddings == save_info.get_num_iterations())
    if num_of_bins_for_tf is None:
        num_of_bins_for_tf = [10]
    elif isinstance(num_of_bins_for_tf, int):
        num_of_bins_for_tf = [num_of_bins_for_tf]

    embedding.train_embedding(graph=graph, save_info=save_info, removed_nodes=[],
                              num_of_embeddings=num_of_embeddings)

    first_started_embedding = save_info.get_list_of_available_embeddings(graph=graph, find_started_trainings=True)

    tested_nodes = utils.sample_low_avg_high_degree_nodes(graph=graph,
                                                          quantity=num_of_test_evaluations_per_degree_level,
                                                          init_range=2, pref_list=first_started_embedding)
    print(f"\nTrain Embeddings for nodes {tested_nodes}")
    nodes_for_training_embedding = {}

    for index, first_node in enumerate(tested_nodes):
        # print(f"Start training embedding for {index}({first_node}). node.")
        graph_removed_one = graph.delete_node(first_node)
        embedding.train_embedding(graph=graph_removed_one, save_info=save_info, removed_nodes=[first_node],
                                  num_of_embeddings=num_of_embeddings)

        if num_of_training_graphs:

            second_completed_diffs = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                  removed_first_node=first_node,
                                                                                  find_started_trainings=False)

            second_started_embedding = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                  removed_first_node=first_node,
                                                                                  find_started_trainings=True)

            second_tested_nodes = utils.sample_randomly_with_pref_list_without_splitting_nodes(
                graph=graph_removed_one, pref_list=second_completed_diffs,
                secondary_pref_list=second_started_embedding,
                all_list=graph_removed_one.nodes(),
                quantity=num_of_training_graphs)
        else:
            second_tested_nodes = graph_removed_one.nodes()

        nodes_for_training_embedding[first_node] = second_tested_nodes

        # print(f"\nTrain embeddings for removed node {first_node} and {second_tested_nodes}")
        for index2, second_node in enumerate(second_tested_nodes):
            # print(f"Start train embedding {index2}({second_node}) for for {index}({first_node}). node.")
            graph_removed_two = graph_removed_one.delete_node(second_node)
            embedding.train_embedding(graph=graph_removed_two, save_info=save_info,
                                      removed_nodes=[first_node, second_node],
                                      num_of_embeddings=num_of_embeddings)

    # create features
    if run_experiments_on_embedding:

        for num_bins in num_of_bins_for_tf:
            # try:
            cf.compute_training_features(save_info=save_info, graph=graph, num_of_bins=num_bins,
                                         list_nodes_to_predict=tested_nodes,
                                         nodes_to_train_on=nodes_for_training_embedding, feature_type=feature_type)
            te.test(save_info=save_info, graph=graph, feature_type=feature_type, num_of_bins=num_bins,
                    limit_num_training_graphs=num_of_training_graphs, list_nodes_to_predict=tested_nodes,
                    nodes_to_train_on=nodes_for_training_embedding)
            # except Exception as e:
            #  print(f"Failed to compute Training Features or Test. "
            #          f"graph {str(graph)}, "
            #          f"emb {str(embedding)}, "
            #          f"num_bins {num_bins}")
            #   traceback.print_exc()

    return tested_nodes, nodes_for_training_embedding
