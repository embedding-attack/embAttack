

import experiments.exp_train_embedding as pe
import memory_access as sl
import embeddings.embedding as abs_emb
import graphs.graph_class as gc
import experiments.exp_calc_features_from_embedding as cf
import classifier.train_sklearn_predictions as te
import features.feature_type as ft
import features.diff_type as dt
from typing import List


def run_experiments(graphs: List[gc.Graph], embeddings: List[abs_emb.Embedding],
                    diff_types: List[dt.DiffType] = [dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE],
                    list_num_tr_graphs: List[int] = [10], list_num_iter: List[int] = [1],
                    list_num_bins: List[int] = [10],
                    feature_type: ft.FeatureType = ft.FeatureType.DIFF_BIN_WITH_DIM, num_test_eval: int = 5,
                    num_eval_iter: int = 1):
    # what to train
    embeddings = list(embeddings)
    for graph in graphs:
        for embedding in embeddings:
            # how to train
            for num_tr_graphs in list_num_tr_graphs:
                num_iterations = max(num_eval_iter, max(list_num_iter))

                save_info = sl.MemoryAccess(graph=str(graph), embedding_type=str(embedding),
                                            num_iterations=num_iterations,
                                            diff_type=dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE)

                tested_nodes, training_nodes = pe.train_embedding_per_graph(graph=graph, embedding=embedding,
                                                                            save_info=save_info,
                                                                            num_of_embeddings=num_iterations,
                                                                            num_of_training_graphs=num_tr_graphs,
                                                                            run_experiments_on_embedding=False,
                                                                            num_of_test_evaluations_per_degree_level=num_test_eval)

            for diff_type in diff_types:
                save_info.set_diff_type(diff_type)
                for num_iter in list_num_iter:
                    save_info.set_num_iter(num_iter)
                    for num_bins in list_num_bins:
                        cf.compute_training_features(save_info=save_info, graph=graph,
                                                     feature_type=feature_type,
                                                     num_of_bins=num_bins,
                                                     list_nodes_to_predict=tested_nodes,
                                                     nodes_to_train_on=training_nodes,
                                                     num_eval_iterations=num_eval_iter)

                        te.test(save_info=save_info, graph=graph, feature_type=feature_type,
                                num_of_bins=num_bins,
                                limit_num_training_graphs=num_tr_graphs, list_nodes_to_predict=tested_nodes,
                                nodes_to_train_on=training_nodes, num_eval_iterations=num_eval_iter)
