import graphs.graph_class as gc
import memory_access as sl
import features.feature_type as ft
import numpy as np
import utils


def filter_by_already_trained_nodes(p_node_list: [int], t_node_dict: {int: [int]}, graph: gc.Graph,
                                    save_info: sl.MemoryAccess, feature_type: ft.FeatureType, num_bins: int):
    '''
    ths function filters test and training features that have already been trained from the list_nodes_to_predict and
    nodes_to_train_on
    :param p_node_list: the list contain all nodes for which training feature should be computed
    :param t_node_dict: dict that contains a mapping from a first node the list of second nodes where training
                              features should be comuted of
    :param save_info: memory management class to access files
    :param feature_type: type of the training featues that should be created
    :param num_bins: number of bins the feature should contain
    :return:
    '''
    np.testing.assert_array_equal(p_node_list, list(t_node_dict.keys()))

    new_nodes_to_train_on = {}
    new_list_nodes_to_predict = []

    for node_to_predict in p_node_list:
        tr_nodes_without_features = list(filter(
            lambda node: not save_info.has_training_data(removed_nodes=[node_to_predict, node],
                                                         feature_type=feature_type,
                                                         num_of_bins=num_bins), t_node_dict[node_to_predict]))

        if len(tr_nodes_without_features) == 0 and save_info.has_training_data(removed_nodes=[node_to_predict],
                                                                               feature_type=feature_type,
                                                                               num_of_bins=num_bins):
            pass
        else:
            new_list_nodes_to_predict.append(node_to_predict)
            new_nodes_to_train_on[node_to_predict] = tr_nodes_without_features

    return new_list_nodes_to_predict, new_nodes_to_train_on


def __get_available_sample(graph: gc.Graph, degrees: [int], center, init_range: int, quantity,
                           available_list: [int],
                           neg_list: [int]) -> []:
    assert (set(available_list).issubset(set(graph.nodes())))

    degrees = np.array(degrees)
    candidates = utils.__get_candidates_with_offset(degrees=degrees, graph=graph, candidate_degree=center,
                                                    neg_list=neg_list)
    offset = 1
    while (offset < init_range) or (len(candidates) < quantity):
        new_candidates = utils.__get_candidates_with_offset(degrees=degrees, graph=graph,
                                                            candidate_degree=center + offset,
                                                            neg_list=neg_list)
        new_candidates += utils.__get_candidates_with_offset(degrees=degrees, graph=graph,
                                                             candidate_degree=center - offset,
                                                             neg_list=neg_list)
        candidates += new_candidates
        offset += 1

    # priorities candidates from pref_list
    pref_candidates = list(set(candidates).intersection(set(available_list)))
    if len(pref_candidates) < quantity:
        raise ValueError(f"Not all nodes available for sampling nodes with about {center} degrees. Grapg {str(graph)}")

    return pref_candidates[:quantity]


def filter_by_splitting_nodes(tr_nodes: [], graph_rem_one: gc.Graph):
    return list(filter(lambda node: not graph_rem_one.splits_graph(node), tr_nodes))


def get_available_graph_data(graph: gc.Graph, save_info: sl.MemoryAccess, num_of_training_graphs: int):
    complete_data = {}

    te_nodes = save_info.get_list_of_available_embeddings(graph=graph, find_started_trainings=False)

    for te_node in te_nodes:
        graph_removed_one = graph.delete_node(te_node)

        second_completed_embeddings = save_info.get_list_of_available_embeddings(graph=graph_removed_one,
                                                                                 removed_first_node=te_node,
                                                                                 find_started_trainings=False)
        second_completed_embeddings = filter_by_splitting_nodes(tr_nodes=second_completed_embeddings,
                                                                graph_rem_one=graph_removed_one)

        if len(second_completed_embeddings) >= num_of_training_graphs:
            complete_data[te_node] = second_completed_embeddings[:num_of_training_graphs]
            # np.random.choice(a=second_completed_embeddings, size=num_of_training_graphs,replace=False)

    return complete_data


def get_min_avg_max_sample_from_available_list(graph: gc.Graph, quantity: int, available_list: [],
                                               init_range: int = 2, ):
    degrees = graph.all_degrees()

    min_val: int = min(degrees)
    max_val: int = max(degrees)
    avg_val: int = int(round(((max_val - min_val) / 2) + min_val))  # int(round(np.array(degrees).mean()))

    max_sample = __get_available_sample(graph=graph, degrees=degrees, center=max_val, init_range=init_range,
                                        quantity=quantity, available_list=available_list, neg_list=[])
    min_sample = __get_available_sample(graph=graph, degrees=degrees, center=min_val, init_range=init_range,
                                        quantity=quantity, available_list=available_list, neg_list=list(max_sample))
    avg_sample = __get_available_sample(graph=graph, degrees=degrees, center=avg_val, init_range=init_range,
                                        quantity=quantity, available_list=available_list,
                                        neg_list=list(max_sample) + list(min_sample))

    # print(f"samles: \n    max {max_sample}\n    min: {min_sample}\n    avg: {avg_sample}")
    samples = np.concatenate((max_sample, avg_sample, min_sample))

    assert (len(set(samples)) == len(samples))

    return samples


def get_nodes_with_trained_embeddings(graph: gc.Graph, save_info: sl.MemoryAccess,
                                      num_of_training_graphs: int,
                                      num_test_eval: int = 5):
    data = get_available_graph_data(graph=graph, save_info=save_info,
                                    num_of_training_graphs=num_of_training_graphs)
    if not (len(data.keys()) >= num_test_eval * 3 and all(
            [len(val) == num_of_training_graphs for val in data.values()])):
        msg = ""
        if not len(data.keys()) >= num_test_eval * 3:
            msg += f"\nBefore Sampling: Number of test nodes is too small. Available {len(data.keys())} " \
                f"required {num_test_eval * 3}."

        for key, val in data.items():
            if len(val) != num_of_training_graphs:
                msg += f"\n Number of training networks for test node {key} is too small. " \
                    f"Available tr networks {val}(len {len(val)}).Should be {num_of_training_graphs}."

        raise ValueError(f"Not all training data is avialable for {save_info}, num_tr_graphs {num_of_training_graphs},"
                         f"num_test_eval {num_test_eval}. Further Information:" + msg)

    samled_nodes = get_min_avg_max_sample_from_available_list(graph=graph, quantity=num_test_eval, init_range=2,
                                                              available_list=data.keys())

    if not all([s_node in data.keys() for s_node in samled_nodes]):
        raise FileNotFoundError(
            f"Enough embeddings found though not in correct degree level! {str(save_info)}, {str(graph)},"
            f"num tr graphs {num_of_training_graphs}, "
            f"num_eval_per_deg_level: {num_test_eval}")

    # filter out data that was not used
    data = dict(filter(lambda i: i[0] in samled_nodes, data.items()))

    if not (len(data.keys()) == num_test_eval * 3 and all(
            [len(val) == num_of_training_graphs for val in data.values()])):
        msg = ""
        if not len(data.keys()) == num_test_eval * 3:
            msg += f"\nAfter Sampling: Number of test nodes is too small. " \
                f"Available {len(data.keys())} required {num_test_eval * 3}."

        for key, val in data.items():
            if len(val) != num_of_training_graphs:
                msg += f"\n Number of training networks for test node {key} is too small. " \
                    f"Available tr networks {val}(len {len(val)}).Should be {num_of_training_graphs}."

        raise ValueError(f"Not all training data is avialable for {save_info}, num_tr_graphs {num_of_training_graphs},"
                         f"num_test_eval {num_test_eval}" + msg)

    return list(data.keys()), data
