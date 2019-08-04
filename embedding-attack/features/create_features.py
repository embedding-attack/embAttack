import pandas as pd

import utils
import memory_access as sl
import graphs.graph_class as gc
import collections as c
from sklearn import preprocessing
from features import diff_type as dt, equisizedbins as bs, feature_type as ft
import embeddings.node2vec_c_path_gensim_emb as n2v
import numpy as np


def __reduce_matrix(dm: pd.DataFrame, rem_node):
    dm_r = dm.drop(rem_node, axis=1)
    return dm_r.drop(rem_node, axis=0)


def reduce_dm(dm_original: pd.DataFrame, rem_node: [int]):
    dm_reduced = dm_original.drop(index=rem_node, columns=rem_node)
    assert (len(dm_original) - 1 == len(dm_reduced) and len(dm_original.columns) - 1 == len(dm_reduced.columns))
    return dm_reduced


def create_difference_matrix(dm_original: pd.DataFrame, dm_reduced: pd.DataFrame,
                             removed_nodes: [int], save_info: sl.MemoryAccess, save: bool = True,
                             check_for_existing: bool = True) -> pd.DataFrame:
    diff_type = save_info.get_diff_type()

    type_to_func = {dt.DiffType.DIFFERENCE: create_difference_matrix_difference,
                    dt.DiffType.DIFFERENCE_ONE_INIT: create_difference_matrix_difference,
                    dt.DiffType.RATIO: create_difference_matrix_ratio}

    return type_to_func[diff_type](dm_original=dm_original, dm_reduced=dm_reduced, removed_nodes=removed_nodes,
                                   save_info=save_info, save=save, check_for_existing=check_for_existing)


def create_difference_matrix_difference(dm_original: pd.DataFrame, dm_reduced: pd.DataFrame,
                                        removed_nodes: [int], save_info: sl.MemoryAccess, save: bool = True,
                                        check_for_existing: bool = True, reduce_o_dm: bool = True) -> pd.DataFrame:

    if check_for_existing and save_info.has_diff_matrix(removed_nodes):
        diff, _ = save_info.load_diff_matrix(removed_nodes)
        return diff

    # reduce original dem to match red dm
    if reduce_o_dm:
        dm_o = reduce_dm(dm_original=dm_original, rem_node=removed_nodes[-1])
    else:
        dm_o = dm_original
    # assert (removed_nodes[-1] not in list(dm_o.index))
    utils.assure_same_labels([dm_o, dm_reduced],
                             f"compute difference matrix: differing Indices! "
                             f"For removed nodes {removed_nodes}\n Save_info: {save_info}")
    diff = dm_o - dm_reduced

    if save:
        save_info.save_diff_matrix(removed_nodes, diff, diff_type=save_info.get_diff_type())
    return diff


def create_difference_matrix_ratio(dm_original: pd.DataFrame, dm_reduced: pd.DataFrame,
                                   removed_nodes: [int], save_info: sl.MemoryAccess, save: bool = True,
                                   check_for_existing: bool = True) -> pd.DataFrame:

    if not save_info.is_diff_type(dt.DiffType.RATIO):
        raise ValueError(f"MemoryAccess object does not specify a difference type. To run this function"
                         f"the diff type must be diff type '{dt.DiffType.DIFFERENCE}'")

    if check_for_existing and save_info.has_diff_matrix(removed_nodes):
        print("difference matrix for removed nodes {} and\
         num iterations {} and type {} already exists!".format(removed_nodes, save_info.num_iterations,
                                                               dt.DiffType.RATIO))
        return save_info.load_diff_matrix(removed_nodes)

    # reduce original dem to match red dm
    dm_o = reduce_dm(dm_original=dm_original, rem_node=list(dm_reduced.index))
    assert (removed_nodes[-1] not in list(dm_o.index))

    # utils.assure_same_labels([dm_o, dm_reduced],
    #                         f"Checking of original distance matrix (removed nodes {removed_nodes[:-1]}) \
    #                                  and reduced distance matrix
    #                                  (removed nodes {removed_nodes}) have the same labels \
    #                                  after removing the last label from the ordginal distance matrix")

    ratio = dm_o / dm_reduced
    if save:
        save_info.save_diff_matrix(removed_nodes, ratio, diff_type=dt.DiffType.RATIO)

    return ratio


def aggregate_node_distance_change(diff: pd.DataFrame):
    utils.assure_same_labels([diff])
    labels = diff.index.values.tolist()
    dim = len(labels)

    node_pos_sums = c.OrderedDict.fromkeys(labels)

    for label in labels:
        node_pos_sums[label] = 0

    for i in range(dim):
        for j in range(i):
            label1 = labels[i]
            label2 = labels[j]

            value = diff.at[label1, label2]
            if value > 0:
                node_pos_sums[label1] += value
                node_pos_sums[label2] += value

    return node_pos_sums


def create_node_raking_from_diff_matrix(diff: pd.DataFrame,
                                        removed_nodes: [int],
                                        graph: gc.Graph,
                                        save_info: sl.MemoryAccess,
                                        save: bool = True) -> []:
    utils.assure_same_labels([diff])

    labels = diff.index.values.tolist()
    dim = len(labels)

    # init sums
    node_pos_sums = {}
    node_neg_sums = {}

    for label in labels:
        node_pos_sums[label] = 0
        node_neg_sums[label] = 0

    # sum values up
    for i in range(dim):
        for j in range(i):
            label1 = labels[i]
            label2 = labels[j]

            value = diff.at[label1, label2]
            if value > 0:
                node_pos_sums[label1] += value
                node_pos_sums[label2] += value
            else:
                node_neg_sums[label1] += value
                node_neg_sums[label2] += value

    pos_list = list(map(lambda x: (x, node_pos_sums[x]), node_pos_sums))
    neg_list = list(map(lambda x: (x, node_neg_sums[x]), node_neg_sums))

    complete_list = list(map(lambda x: (x, node_pos_sums[x] - node_neg_sums[x]), node_pos_sums))

    pos_list.sort(key=lambda x: -x[1])
    neg_list.sort(key=lambda x: x[1])
    complete_list.sort(key=lambda x: -x[1])

    if save:
        save_info.save_node_raking(removed_nodes, pos_list, list(graph.neighbours(removed_nodes[-1])))

    neighbours = list(graph.neighbours(removed_nodes[0]))
    pos_list_labels = list(map(lambda x: x[0] in neighbours, pos_list))

    neg_list_labels = list(map(lambda x: x[0] in neighbours, neg_list))
    complete_list_labels = list(map(lambda x: x[0] in neighbours, complete_list))

    return pos_list, pos_list_labels, neg_list, neg_list_labels, complete_list, complete_list_labels


def create_target_vector(row_labels: [], graph: gc.Graph, node_to_predict: int) -> pd.DataFrame:
    """
    creates the target vector for classifier
    :param row_labels: labels the target vector should be created
    :param graph: graph including the removed node
    :param node_to_predict: the node that is removed in the 2. embedding
    :return:
    """

    neighbours_of_removed_node = graph.neighbours(node_to_predict)

    target = pd.DataFrame(False, row_labels, ["y"])

    for neighbour in neighbours_of_removed_node:

        # this prevents an error in case the original graph is used while 2 nodes are removed in the labels
        # and they are connected
        if neighbour in row_labels:
            target.loc[neighbour] = True

    return target


def get_feature_from_bin(label_of_feature: int, bins: bs.EquisizedBins, diff: pd.DataFrame):
    feature = [0] * bins.get_number_of_bins()
    labels = utils.get_row_labels(diff)
    for other_label in labels:
        if other_label == label_of_feature:
            continue
        category = bins.get_category(utils.get_difference(label_of_feature, other_label, diff))
        feature[category] += 1

    # normalise
    feature = list(map(lambda x: float(x) / len(labels), feature))

    return feature


def get_features_from_bins(diff: pd.DataFrame, num_of_bins: int):
    bins: bs.EquisizedBins = bs.EquisizedBins(num_of_bins, diff)
    labels = utils.get_row_labels(diff)
    features = pd.DataFrame(0, labels, ["bin " + str(i) for i in range(bins.get_number_of_bins())])

    for label in labels:
        features.loc[label] = get_feature_from_bin(label, bins, diff)

    return features


def create_feature_from_diff_bins(diff: pd.DataFrame, removed_nodes: [int],
                                  original_graph: gc.Graph, num_of_bins: int,
                                  save_info: sl.MemoryAccess, save: bool = True):
    target = create_target_vector(utils.get_row_labels(diff), original_graph, removed_nodes[-1])

    # calculate bin distribution for all labels
    features = get_features_from_bins(diff=diff, num_of_bins=num_of_bins)

    if save:
        save_info.save_training_data(removed_nodes, feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM,
                                     num_of_bins=num_of_bins,
                                     training_data=utils.pd_append_column(features, target))

    return features, target


def __create_degree_column_for_feature(graph: gc.Graph, row_labels: [int]):
    degrees = pd.DataFrame(0, row_labels, ["degree"])
    for label in row_labels:
        degrees.loc[label] = graph.degree(label)

    # normalise degrees
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(degrees)

    degrees = degrees / max(degrees.values)

    return pd.DataFrame(degrees, index=row_labels, columns=["degree"])




def two_hop_neighbours(nodes, graph, node_to_predict):
    """
        creates the target vector for classifier while all nodes within a distance of 2 are labeld as true
        :param nodes: labels the target vector should be created
        :param graph: graph including the removed node
        :param node_to_predict: the node that is removed in the 2. embedding
        :return:
        """

    neighbours_of_removed_node = graph.two_hop_neighbours(node_to_predict)

    target = pd.DataFrame(False, nodes, ["y"])

    for neighbour in neighbours_of_removed_node:
        # this prevents an error in case the original graph is used while 2 nodes are removed in the labels
        # and they are connected
        if neighbour in nodes:
            target.loc[neighbour] = True

    return target


def create_features(diff: pd.DataFrame, removed_nodes: [int], original_graph: gc.Graph,
                    num_of_bins, feature_type: ft.FeatureType,
                    save_info: sl.MemoryAccess, save: bool = True, output_feature: bool = False,
                    check_for_existing: bool = True):
    if feature_type == ft.FeatureType.DIFF_BIN_WITH_DIM:
        return create_feature_from_diff_bins_with_dim(diff=diff, removed_nodes=removed_nodes,
                                                      original_graph=original_graph,
                                                      num_of_bins=num_of_bins,
                                                      save_info=save_info, save=save, output_feature=output_feature,
                                                      check_for_existing=check_for_existing)
    elif feature_type == ft.FeatureType.DIFF_BIN_WITH_DIM_2_HOP:
        features, target = create_feature_from_diff_bins_with_dim(diff=diff, removed_nodes=removed_nodes,
                                                                  original_graph=original_graph,
                                                                  num_of_bins=num_of_bins,
                                                                  save_info=save_info, save=False, output_feature=True)
        target = two_hop_neighbours(nodes=utils.get_row_labels(features), graph=original_graph,
                                    node_to_predict=removed_nodes[-1])
        if save:
            save_info.save_training_data(removed_nodes=removed_nodes,
                                         feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM_2_HOP, num_of_bins=num_of_bins,
                                         training_data=utils.pd_append_column(features, target))

        return features, target
    elif feature_type == ft.FeatureType.EVEN_DIST:
        raise NotImplementedError()

    else:
        raise ValueError(f"Feature type {feature_type} is not known!")


def compute_degrees(graph: gc.Graph, labels: [int]):
    degrees = pd.DataFrame(0, labels, ["degree"])
    for label in labels:
        degrees.loc[label] = graph.degree(label)

    # normalise degrees
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(degrees.astype(float))

    return pd.DataFrame(x_scaled, index=labels, columns=["degree"])


def create_feature_from_diff_bins_with_dim(diff: pd.DataFrame, removed_nodes: [int],
                                           original_graph: gc.Graph, num_of_bins, save_info: sl.MemoryAccess,
                                           save: bool = True, output_feature: bool = False,
                                           check_for_existing: bool = True):
    if check_for_existing and save_info.has_training_data(removed_nodes=removed_nodes,
                                                          feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM,
                                                          num_of_bins=num_of_bins):
        # print("training feature for removed nodes '{}' and feature type '{}' already exists"
        #      .format(removed_nodes, ft.FeatureType.DIFF_BIN_WITH_DIM.to_str(num_of_bins)))
        if output_feature:
            data = save_info.load_training_data(removed_nodes=removed_nodes,
                                                feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM, num_of_bins=num_of_bins)
            features = data.drop(["y"], axis=1)
            labels = data["y"]
            return features, labels
        else:
            return

    features, target = create_feature_from_diff_bins(diff=diff, removed_nodes=removed_nodes,
                                                     original_graph=original_graph, num_of_bins=num_of_bins,
                                                     save_info=save_info, save=False)

    labels = utils.get_row_labels(diff)
    degrees = compute_degrees(graph=original_graph, labels=labels)

    features = utils.pd_append_column(features, degrees)

    if save:
        save_info.save_training_data(removed_nodes, feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM,
                                     num_of_bins=num_of_bins,
                                     training_data=utils.pd_append_column(features, target))
    if output_feature:
        # output format is different for data loaded from file!
        return features, target


def main():
    graph = gc.Graph.init_karate_club_graph()
    graph = gc.Graph.init_sampled_aps_pacs052030()
    embedding_function = n2v.Node2VecPathSnapEmbGensim()

    num_of_bins = 10
    num_iterations = 30
    # nodes_to_predict = list(graph.nodes)
    save_info = sl.MemoryAccess(graph=str(graph), embedding_type=str(embedding_function),
                                num_iterations=num_iterations)

    for i in range(34):
        removed_nodes = [i]
        diff = save_info.load_diff_matrix(removed_nodes)
        create_feature_from_diff_bins_with_dim(diff=diff, removed_nodes=removed_nodes,
                                               original_graph=graph, num_of_bins=num_of_bins,
                                               save_info=save_info, save=True, check_for_existing=False)

        for j in range(34):
            if i != j:
                removed_nodes = [i, j]

                diff = save_info.load_diff_matrix(removed_nodes)
                create_feature_from_diff_bins_with_dim(diff=diff, removed_nodes=removed_nodes,
                                                       original_graph=graph, num_of_bins=num_of_bins,
                                                       save_info=save_info, save=True, check_for_existing=False)

    # train.test(save_info=save_info, graph=graph, feature_type=ft.FeatureType.DIFF_BIN_WITH_DIM, num_of_bins=num_of_bins)


if __name__ == '__main__':
    main()
