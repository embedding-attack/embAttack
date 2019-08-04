from typing import List
import warnings

import pandas as pd
import graphs.graph_class as gc
import numpy as np


def assure_same_labels(data_frames: List[pd.DataFrame], info_text: str = "") -> None:
    labels = data_frames[0].columns.values.tolist()

    if not all(
            [dm.columns.values.tolist() == labels for dm in data_frames] + [dm.index.values.tolist() == labels for dm in
                                                                            data_frames]):
        set_labels = set(labels)
        msg = ""
        for index, dm in enumerate(data_frames):
            set_i = set(dm.index.values.tolist())
            set_c = set(dm.columns.values.tolist())
            # print("Dataframe ", index, " labels:", dm.columns.values.tolist(),dm.index.values.tolist())


            msg += f"\n Dataframe {index}: Differences to f1.columns: " \
                f"\n Value in index: {set_i - set(labels)}, Value not in Index {set(labels) - set_i}," \
                f"\n Value in column: {set_c - set(labels)}, Value not in column {set(labels) - set_c},"

        raise ValueError(
            "One of the dataframes does not have the correct structure. "
            "Hint: If the dataframes are loaded it might be due to a "
            "file that does not have the correct format.\n" + info_text + msg)


def get_difference(label1, label2, diff: pd.DataFrame):
    return diff.at[label1, label2]
    if label1 > label2:
        return diff.at[label1, label2]
    else:
        return diff.at[label2, label1]


def get_row_labels(data_frame: pd.DataFrame):
    return data_frame.index.values.tolist()


def pd_append_column(df1: pd.DataFrame, df2: pd.DataFrame):
    if get_row_labels(df1) != get_row_labels(df2):
        raise ValueError(
            'Row names differ! Make sure that the row names have the same dimensionality and names \n \
                row labels of Dataframe 1:'
            + str(get_row_labels(df1)) + "\n row labels for Dataframe 2:" + str(get_row_labels(df2)))

    return df1.join(df2)


def __get_graph_degree_properties(graph: gc.Graph):
    degrees = graph.all_degrees()
    min_val = min(degrees)
    max_val = max(degrees)
    avg = np.mean(degrees)
    return min_val, max_val, avg


def __get_nodes_in_range(nodes: [int], degrees: [int], deg_range):
    assert (len(degrees) == len(nodes))

    res = []
    for i in range(len(nodes)):
        if degrees[i] in deg_range:
            res.append(nodes[i])

    return res


def __get_range_around_number(number: int, offset: int):
    return range(number - offset, number + offset)


def get_sample_with_degree(graph: gc.Graph, node_list: [int], degree: int, quantity: int):
    degrees = np.array([graph.degree(n) for n in node_list])
    samples_to_find = quantity

    candidates = np.where(degrees == degree)[0]
    if len(candidates) > samples_to_find:
        np.random.choice(candidates, size=samples_to_find, replace=False)
    elif len(candidates) < samples_to_find:
        raise ValueError(f'Not enough training samples required {quantity} got {len(candidates)}')
    sample = candidates
    samples_to_find -= len(candidates)

    offset = 1
    while samples_to_find > 0:
        # print(f"Sampling range {offset}")
        candidates = np.concatenate([np.where(degrees == (degree + offset))[0],
                                     np.where(degrees == (degree - offset))[0]])
        if len(candidates) > samples_to_find:
            candidates = np.random.choice(candidates, size=samples_to_find, replace=False)
        sample = np.concatenate([sample, candidates])
        samples_to_find -= len(candidates)
        offset += 1

    return [node_list[s] for s in sample]


def __get_candidates_with_offset(degrees: np.ndarray, graph: gc.Graph, candidate_degree: int, neg_list: List[int]):
    indices = np.where(degrees == candidate_degree)[0].tolist()
    new_candidates = [graph.nodes()[i] for i in indices]
    new_candidates = list(filter(lambda x: x not in neg_list, new_candidates))
    new_candidates = __filter_splitting_nodes(node_list=new_candidates, graph=graph)
    return new_candidates


def __get_sample(graph: gc.Graph, degrees: [int], center, init_range: int, quantity, pref_list: [int],
                 neg_list: [int]) -> np.ndarray:
    assert (set(pref_list).issubset(set(graph.nodes())))

    degrees = np.array(degrees)
    candidates = __get_candidates_with_offset(degrees=degrees, graph=graph, candidate_degree=center, neg_list=neg_list)
    offset = 1
    while (offset < init_range) or (len(candidates) < quantity):
        new_candidates = __get_candidates_with_offset(degrees=degrees, graph=graph, candidate_degree=center + offset,
                                                      neg_list=neg_list)
        new_candidates += __get_candidates_with_offset(degrees=degrees, graph=graph, candidate_degree=center - offset,
                                                       neg_list=neg_list)
        candidates += new_candidates
        offset += 1

    # priorities candidates from pref_list
    pref_candidates = list(set(candidates).intersection(set(pref_list)))
    return sample_randomly_with_preferred_list(pref_list=pref_candidates, all_list=candidates, quantity=quantity)


def sample_low_avg_high_degree_nodes(graph: gc.Graph, quantity: int, init_range: int = 2, pref_list=None):
    if pref_list is None:
        pref_list = []
    degrees = graph.all_degrees()

    min_val: int = min(degrees)
    max_val: int = max(degrees)
    avg_val: int = int(round(((max_val - min_val) / 2) + min_val))  # int(round(np.array(degrees).mean()))

    nodes = graph.nodes()

    max_sample = __get_sample(graph=graph, degrees=degrees, center=max_val, init_range=init_range, quantity=quantity,
                              pref_list=pref_list, neg_list=[])
    min_sample = __get_sample(graph=graph, degrees=degrees, center=min_val, init_range=init_range, quantity=quantity,
                              pref_list=pref_list, neg_list=list(max_sample))
    avg_sample = __get_sample(graph=graph, degrees=degrees, center=avg_val, init_range=init_range, quantity=quantity,
                              pref_list=pref_list, neg_list=list(max_sample) + list(min_sample))

    # print(f"samles: \n    max {max_sample}\n    min: {min_sample}\n    avg: {avg_sample}")
    samples = np.concatenate((max_sample, avg_sample, min_sample))

    assert (len(set(samples)) == len(samples))

    return samples


def __filter_splitting_nodes(node_list: List[int], graph: gc.Graph):
    return list(filter(lambda x: not graph.splits_graph(x), node_list))


def __get_filtered_random_nodes(all_other_list: List[int], num_needed_nodes: int, graph: gc.Graph):
    candidates = np.random.choice(a=all_other_list, size=num_needed_nodes, replace=False)

    target_list = __filter_splitting_nodes(node_list=candidates, graph=graph)
    num_needed_nodes = num_needed_nodes - len(target_list)

    if num_needed_nodes > 0:
        all_other_list = list(filter(lambda elem: elem not in candidates, all_other_list))
        target_list += __get_filtered_random_nodes(all_other_list=all_other_list, num_needed_nodes=num_needed_nodes,
                                                   graph=graph)

    return target_list


def sample_randomly_with_pref_list_without_splitting_nodes(graph: gc.Graph, pref_list: [int],
                                                           all_list: [int], quantity: int,
                                                           secondary_pref_list: [int] = None):
    if not set(pref_list).issubset(set(secondary_pref_list)):
        raise ValueError(f"preflist is not part of secondary pref list"
                         f"\n pref list: {pref_list}\n secondary pref list: {secondary_pref_list}")

    if not set(secondary_pref_list).issubset(set(all_list)):
        raise ValueError(f"secondary pref list is not part of all list"
                         f"\n secondary pref list: {secondary_pref_list}\n all list: {all_list}")

    pref_list = __filter_splitting_nodes(node_list=pref_list, graph=graph)
    secondary_pref_list = __filter_splitting_nodes(node_list=secondary_pref_list, graph=graph)

    guaranteed_list = []
    all_other_list: List[int]

    if len(pref_list) >= quantity:
        all_other_list = pref_list
    elif len(secondary_pref_list) >= quantity:
        guaranteed_list = pref_list
        all_other_list = list(filter(lambda elem: elem not in pref_list, secondary_pref_list))
    else:
        guaranteed_list = secondary_pref_list
        all_other_list = list(filter(lambda elem: elem not in secondary_pref_list, all_list))

    target_list = __filter_splitting_nodes(node_list=guaranteed_list, graph=graph)
    target_list += __get_filtered_random_nodes(all_other_list=all_other_list,
                                               num_needed_nodes=quantity - len(target_list), graph=graph)
    return target_list


def sample_randomly_with_preferred_list(pref_list: [int], all_list: [int], quantity: int,
                                        secondary_pref_list: [int] = None):
    assert (set(pref_list).issubset(set(all_list)))

    if secondary_pref_list is None:
        if len(pref_list) >= quantity:
            return pref_list[:quantity] #np.random.choice(pref_list, size=quantity, replace=False)
        else:
            all_others = list(filter(lambda elem: elem not in pref_list, all_list))
            return pref_list + list(np.random.choice(all_others, size=quantity - len(pref_list), replace=False))

    else:
        assert (set(secondary_pref_list).issubset(set(all_list)))
        if not set(secondary_pref_list).issuperset(set(pref_list)):
            warnings.warn(
                f"Secondary pref does not contain pref_list \n"
                f"Secondary pref list:{secondary_pref_list}\nPref list:{pref_list}")
            secondary_pref_list = list(set(secondary_pref_list).union(set(pref_list)))

        if len(pref_list) >= quantity:
            return np.random.choice(pref_list, size=quantity, replace=False)
        elif len(secondary_pref_list) >= quantity:
            all_others = list(filter(lambda elem: elem not in pref_list, secondary_pref_list))
            return pref_list + list(np.random.choice(all_others, size=quantity - len(pref_list), replace=False))
        else:
            all_others = list(filter(lambda elem: elem not in secondary_pref_list, all_list))
            return secondary_pref_list + list(
                np.random.choice(all_others, size=quantity - len(secondary_pref_list), replace=False))


def assert_df_no_nan(df: pd.DataFrame, text: str):
    if df.isnull().any().any():
        raise ValueError(f"Dataframe contains nan values. {text}")


