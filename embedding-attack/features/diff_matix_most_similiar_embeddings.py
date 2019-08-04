import pandas as pd

import memory_access as sl
from typing import List
import features.create_features as cf
import numpy as np
import distance_matrices.calculate_distance_matrix as cdm


def compute_diff_size(diff: pd.DataFrame):
    return diff.abs().values.sum()


def compute_diff_matrix_helper(o_dm_list: List[pd.DataFrame], r_dm_list: List[pd.DataFrame],
                               removed_nodes: List[int], save_info: sl.MemoryAccess, save: bool = True,
                               check_for_existing: bool = True):
    # check for existing
    if check_for_existing and save_info.has_diff_matrix(removed_nodes=removed_nodes):
        return save_info.load_diff_matrix(removed_nodes)

    min_diff_size = np.inf
    min_diff: pd.DataFrame
    min_r_dm: pd.DataFrame
    min_r_dm_index: int
    r_dm_list = list(r_dm_list)  # must be called multiple times hence a generator can not be used
    for o_dm in o_dm_list:
        o_dm_reduced = cf.reduce_dm(dm_original=o_dm, rem_node=removed_nodes[-1])
        for index, r_dm in enumerate(r_dm_list):

            diff = cf.create_difference_matrix_difference(dm_original=o_dm_reduced, dm_reduced=r_dm,
                                                          removed_nodes=removed_nodes,
                                                          save_info=save_info, save=False, check_for_existing=False,
                                                          reduce_o_dm=False)
            diff_size = compute_diff_size(diff)
            if diff_size < min_diff_size:
                min_diff = diff
                min_diff_size = diff_size
                min_r_dm = r_dm
                min_r_dm_index = index

    if save:
        save_info.save_diff_matrix(removed_nodes=removed_nodes, diff=min_diff,
                                   diff_type=save_info.diff_type, r_dm_index=min_r_dm_index)
    return min_diff, min_r_dm


def load_dms(removed_nodes: List[int], save_info: sl.MemoryAccess, num_iterations: int, use_specific_iter: int = None):
    if num_iterations == 1:
        assert (use_specific_iter is not None)
        if save_info.has_distance_matrix(removed_nodes=removed_nodes, iteration=use_specific_iter):
            yield save_info.load_distance_matrix(removed_nodes=removed_nodes, iteration=use_specific_iter)
        else:
            emb = save_info.load_embedding(removed_nodes=removed_nodes, iteration=use_specific_iter)
            dm = cdm.calc_distances(model=emb, save_info=save_info, removed_nodes=removed_nodes,
                                    iteration=use_specific_iter)
            yield dm
    else:
        for i in range(num_iterations):
            if save_info.has_distance_matrix(removed_nodes=removed_nodes, iteration=i):
                yield save_info.load_distance_matrix(removed_nodes=removed_nodes, iteration=i)
            else:
                emb = save_info.load_embedding(removed_nodes=removed_nodes, iteration=i)
                dm = cdm.calc_distances(model=emb, save_info=save_info, removed_nodes=removed_nodes, iteration=i)
                yield dm


def compute_diff_matrix(removed_nodes: List[int], save_info: sl.MemoryAccess, quantity_first: int,
                        quantity_second: int,
                        o_dm_list: List[pd.DataFrame] = None, r_dm_list: List[pd.DataFrame] = None,
                        save: bool = True, check_for_existing: bool = True, used_emb: int = None):
    if quantity_first == 1 or quantity_second == 1:
        if used_emb is None or used_emb is -1:
            raise ValueError(
                f"For the used difference type an iteration must be seleted, to chose with embedding should"
                f"be used. Try diff_type.set_iter(n).")

    if o_dm_list is None:
        o_dm_list = load_dms(removed_nodes=removed_nodes[:-1], save_info=save_info, num_iterations=quantity_first,
                             use_specific_iter=used_emb)
    else:
        assert (len(o_dm_list) == quantity_first)

    if r_dm_list is None:
        r_dm_list = load_dms(removed_nodes=removed_nodes, save_info=save_info, num_iterations=quantity_second,
                             use_specific_iter=used_emb)

    return compute_diff_matrix_helper(o_dm_list=o_dm_list, r_dm_list=r_dm_list, removed_nodes=removed_nodes,
                                      save_info=save_info, save=save, check_for_existing=check_for_existing)
