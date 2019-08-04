import glob

import pandas as pd
import numpy as np
import os.path
import gensim
import typing
import deprecated
import graphs.graph_class as gc
import config as config
import features.diff_type as dt
import features.feature_type as ft
import utils as utils


class MemoryAccess:
    def __init__(self, graph: str, embedding_type: str, num_iterations: int, diff_type: dt.DiffType = None):

        self.graph = graph
        self.embedding_type = embedding_type
        self.num_iterations = num_iterations
        self.diff_type = diff_type

        # print('Base directory path used:', base_dir)

        self.GRAPH_BASE_PATH = config.DIR_PATH + f"results/graph_name-{graph}/"
        self.BASE_PATH = self.GRAPH_BASE_PATH + f"embedding_type-{embedding_type}/"
        self.EMBEDDING_PATH = self.BASE_PATH + "embeddings/"
        self.EMBEDDING_PATH = self.BASE_PATH + "embeddings/"
        self.MODEL_PATH = self.BASE_PATH + "model/"
        self.GRAPH_PATH = self.GRAPH_BASE_PATH + "graphs/"
        self.DISTANCE_MATRICES_PATH = self.BASE_PATH + "dms/"
        self.AVG_DISTANCE_MATRICES_PATH = self.BASE_PATH + "avg_dms/"
        self.DIFFERENCE_MATRIX_PATH = self.BASE_PATH + "diffs{}/"
        self.NODE_RANKING_PATH = self.BASE_PATH + "node_rankings/"
        self.PLOTS_PATH = self.BASE_PATH + "plots/"
        self.TRAINING_DATA_PATH = self.BASE_PATH + "training_data/"
        self.TEST_RESULTS_PATH = self.BASE_PATH + "test_results/"
        self.CONTINUED = "_continued"

    def __str__(self):
        return f"{self.graph}_{self.embedding_type}_num_iter_{self.num_iterations}_{self.diff_type}"

    @staticmethod
    def __assure_path(path: str) -> str:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def __get_embedding_search_string(self, emb_description: str = None):
        search_string = f"emb_removed_nodes_\[([0-9]+)\]_iteration{self.num_iterations - 1}"
        if emb_description:
            search_string += glob.escape(f"_addInfo={emb_description}")
        return search_string

    def __get_embedding_name(self, removed_nodes: [int], iteration: int, emb_description: str = None):
        if emb_description is None:
            return self.__assure_path(self.EMBEDDING_PATH) + "emb_removed_nodes_{}_iteration{}".format(
                sorted(removed_nodes), iteration)
        else:
            return self.__assure_path(self.EMBEDDING_PATH) + "emb_removed_nodes_{}_iteration{}_addInfo={}".format(
                sorted(removed_nodes), iteration, emb_description)

    def __get_model_name(self, removed_nodes: [int], iteration: int, emb_description: str = None):
        if emb_description is None:
            return self.__assure_path(self.MODEL_PATH) + "model_removed_nodes_{}_iteration{}.model".format(
                sorted(removed_nodes), iteration)
        else:
            return self.__assure_path(self.MODEL_PATH) + "model_removed_nodes_{}_iteration{}_addInfo={}.model".format(
                sorted(removed_nodes), iteration, emb_description)

    def __get_distance_matrices_name(self, removed_nodes: [int], iteration: int, add_info: str = None):
        if add_info is None:
            add_info = ""

        return self.__assure_path(self.DISTANCE_MATRICES_PATH) + "dm_removed_nodes_{}_iteration_{}{}".format(
            sorted(removed_nodes), iteration, add_info)

    def __get_avg_distance_matrices_name(self, removed_nodes: [int], add_info: str = None):
        if add_info is None:
            add_info = ""
        return self.__assure_path(
            self.AVG_DISTANCE_MATRICES_PATH) + "avg_dm_removed_nodes_{}_num_of_dms{}{}".format(
            sorted(removed_nodes), self.num_iterations, add_info)

    def __get_diff_matrix_name(self, removed_nodes: [int], add_info: str = None, assure_path: bool = True):
        if add_info is None:
            add_info = ""

        path = self.DIFFERENCE_MATRIX_PATH.format(self.get_diff_type())
        if assure_path:
            path = self.__assure_path(path)
        return path + "diff_removed_nodes_{}_num_of_dms_{}{}".format(
            removed_nodes, self.num_iterations, add_info)

    def __get_node_ranking_name(self, removed_nodes: [int], ranking_description: str = None):
        if ranking_description is None:
            return self.__assure_path(self.NODE_RANKING_PATH) + "node_ranking_removed_nodes_{}_num_of_dms{}".format(
                removed_nodes, self.num_iterations)
        else:
            return self.__assure_path(
                self.NODE_RANKING_PATH) + "node_ranking_removed_nodes_{}_num_of_dms{}_description_{}".format(
                removed_nodes, self.num_iterations, ranking_description)

    def __get_plot_name(self, plot_name):
        return self.__assure_path(self.PLOTS_PATH) + "plot_{}.png".format(plot_name)

    def __get_training_data_data_name(self, removed_nodes: [int], feature_type: ft.FeatureType, num_of_bins: int,
                                      add_info: str = None):

        if add_info is None:
            add_info = ""

        return self.__assure_path(
            self.TRAINING_DATA_PATH) + "train_feature_type_{}_removed_nodes_{}_num_of_embeddings{}{}{}".format(
            feature_type.to_str(num_of_bins), removed_nodes, self.num_iterations, self.get_diff_type(),
            add_info)

    def __get_training_data_data_name_iter_30(self, removed_nodes: [int], feature_type: ft.FeatureType,
                                              num_of_bins: int,
                                              add_info: str = None):

        if add_info is None:
            add_info = ""

        return self.__assure_path(
            self.TRAINING_DATA_PATH) + "train_feature_type_{}_removed_nodes_{}_num_of_embeddings{}{}{}".format(
            feature_type.to_str(num_of_bins), removed_nodes, 30, self.get_diff_type(),
            add_info)

    def __get_test_results_name(self, name, assure_path: bool = True):

        if assure_path:
            return self.__assure_path(self.TEST_RESULTS_PATH) + name + ".csv"
        else:
            return self.TEST_RESULTS_PATH + name + ".csv"

    def __get_graph_name(self, removed_nodes: [int], additional_info: str = None):
        if additional_info is None:
            return self.__assure_path(self.GRAPH_PATH) + "edge_list_removed_nodes_{}".format(sorted(removed_nodes))
        else:
            return self.__assure_path(self.GRAPH_PATH) + "edge_list_removed_nodes_{}_addInfo={}".format(
                sorted(removed_nodes), additional_info)

    @staticmethod
    def __save_csv(file_name: str, csv: pd.DataFrame) -> None:
        try:
            csv.to_pickle(file_name + ".csv.gz", compression="gzip")
        except Exception as e:
            # remove file on interrupt to prevent corrupted files
            if os.path.exists(file_name + ".csv.gz"):
                os.remove(file_name + ".csv.gz")
            raise e

    @staticmethod
    def __load_csv(file_name: str) -> pd.DataFrame:
        try:
            csv: pd.DataFrame = pd.read_pickle(file_name + ".csv.gz", compression="gzip")
        except Exception as e:
            import sys
            raise type(e)(f"{str(e)}\n Could not load file with filename: '{file_name}.csv.gz'\n {e}").with_traceback(
                sys.exc_info()[2])
        if "Unnamed: 0" in csv:
            print("csv has wrong format ... correcting it")
            old_indices = csv.index.values
            new_indices = csv["Unnamed: 0"]
            assert (len(old_indices) == len(new_indices))
            rename_map = {}
            for i in range(len(new_indices)):
                rename_map[old_indices[i]] = new_indices[i]
            csv.rename(rename_map, inplace=True)
            csv.drop(columns="Unnamed: 0", inplace=True)
            MemoryAccess.__save_csv(file_name, csv)
        return csv

    @staticmethod
    def __has_csv(file_name: str) -> bool:
        return os.path.exists(file_name + ".csv.gz")

    @staticmethod
    def __has_embedding(file_name: str) -> bool:
        return os.path.exists(file_name + ".csv.gz") or os.path.exists(file_name + ".emb")

    @staticmethod
    def __remove_csv(file_name: str) -> None:
        os.remove(file_name + ".csv.gz")

    # --- public ---

    # ---gettter---
    def get_num_iterations(self):
        return self.num_iterations

    def has_diff_type(self):
        return self.diff_type is not None

    def get_diff_type(self):
        if not self.has_diff_type():
            raise ValueError("Difference Type is not specified!")
        return self.diff_type

    # --- memory access ---

    def set_num_iter(self, num_iterations: int):
        self.num_iterations = num_iterations

    def set_diff_type(self, diff_type: dt.DiffType):
        self.diff_type = diff_type

    def load_custom_embedding(self, name: str):
        path = self.EMBEDDING_PATH + name
        return pd.read_csv(path, index_col=0)

    def save_embedding(self, removed_nodes: [int], iteration: int, embedding, emb_description: str = None):
        if type(embedding) == pd.DataFrame:
            self.__save_csv(file_name=self.__get_embedding_name(removed_nodes, iteration, emb_description),
                            csv=embedding)
        else:
            embedding.wv.save_word2vec_format(
                self.__get_embedding_name(removed_nodes, iteration, emb_description) + ".emb", binary=True)

    def get_embedding_name(self, removed_nodes: [int], iteration: int, emb_description: str = None):
        return self.__get_embedding_name(removed_nodes=removed_nodes, iteration=iteration,
                                         emb_description=emb_description)

    def get_graph_name(self, removed_nodes: [int], additional_info: str = None):
        return self.__get_graph_name(removed_nodes, additional_info)

    def has_embedding(self, removed_nodes: [int], iteration: int, emb_description: str = None):
        return MemoryAccess.__has_embedding(self.__get_embedding_name(removed_nodes, iteration, emb_description))

    @staticmethod
    def parse_emb_to_df(file) -> pd.DataFrame:
        raise NotImplementedError()

    def load_embedding_to_df(self, removed_nodes: [int], iteration: int, node_names: [int], emb_description=None):
        emb_name = self.__get_embedding_name(removed_nodes, iteration, emb_description)
        if os.path.exists(emb_name + ".emb"):
            emb = MemoryAccess.parse_emb_to_df(emb_name + ".emb")
        elif os.path.exists(emb_name + ".csv.gz"):
            emb = self.__load_csv(emb_name)
        else:
            raise FileNotFoundError(f"Embeddingfile \n'{emb_name}'\n exists neither in .emb nor in .csv.cz format!")
        return emb

    def load_embedding(self, removed_nodes: [int], iteration: int, emb_description=None):
        emb_name = self.__get_embedding_name(removed_nodes, iteration, emb_description)

        if os.path.exists(emb_name + ".emb"):
            try:
                emb = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format(emb_name + ".emb", binary=True)
            except Exception as e:
                raise type(e)(f"Failed to load emb for node {removed_nodes} and iteration {iteration}\
                \n Path:{emb_name + '.emb'}\
                \n Error message loading binary: {str(e)}")
            return emb

        elif os.path.exists(emb_name + ".csv.gz"):
            emb = MemoryAccess.__load_csv(emb_name)
            return emb
        else:
            raise ValueError(f"Embedding at path '{emb_name}' does not exist!")

    def remove_embedding(self, removed_nodes: [int], iteration: int, node_names: [int], emb_description=None):
        os.remove(self.__get_embedding_name(removed_nodes, iteration, emb_description) + ".emb")

    @deprecated.deprecated()
    def save_model(self, removed_nodes: [int], iteration: int, model, emb_description: str = None) -> None:
        model.save(self.__get_model_name(removed_nodes, iteration, emb_description))

    @deprecated.deprecated()
    def load_word2vec_model(self, removed_nodes: [int], iteration: int,
                            emb_description: str = None) -> gensim.models.Word2Vec:
        model_path = self.__get_model_name(removed_nodes, iteration, emb_description)
        return gensim.models.Word2Vec.load(model_path)

    def save_distance_matrix(self, removed_nodes: [int], iteration: int, dm: pd.DataFrame,
                             add_info: str = None) -> None:
        MemoryAccess.__save_csv(self.__get_distance_matrices_name(removed_nodes, iteration, add_info=add_info), dm)

    def load_distance_matrix(self, removed_nodes: [int], iteration: int, add_info: str = None) -> pd.DataFrame:
        return MemoryAccess.__load_csv(self.__get_distance_matrices_name(removed_nodes, iteration, add_info=add_info))

    def delete_distance_matrix(self, removed_nodes: [int], iteration: int, add_info: str = None) -> None:
        MemoryAccess.__remove_csv(self.__get_distance_matrices_name(removed_nodes, iteration, add_info=add_info))

    def delete_distance_matrices(self, removed_nodes: [int]):
        if not self.has_avg_distance_matrix(removed_nodes=removed_nodes):
            raise ValueError("Are you sure that you want to delete the dms even though no avg_dm is trained")
        else:
            for i in range(self.num_iterations):
                path = self.__get_distance_matrices_name(removed_nodes=removed_nodes, iteration=i)
                if os.path.exists(path):
                    os.remove(path)

    def has_distance_matrix(self, removed_nodes: [int], iteration: int, add_info: str = None) -> bool:
        return MemoryAccess.__has_csv(self.__get_distance_matrices_name(removed_nodes, iteration, add_info=add_info))

    def remove_distance_matrix(self, removed_nodes: [int], iteration: int, add_info: str = None):
        MemoryAccess.__remove_csv(self.__get_distance_matrices_name(removed_nodes, iteration, add_info=add_info))

    def save_avg_distance_matrix(self, removed_nodes: [int], avg_dm: pd.DataFrame, add_info: str = None) -> None:
        MemoryAccess.__save_csv(self.__get_avg_distance_matrices_name(removed_nodes, add_info=add_info), avg_dm)

    def load_avg_distance_matrix(self, removed_nodes: [int], add_info: str = None) -> pd.DataFrame:
        return MemoryAccess.__load_csv(self.__get_avg_distance_matrices_name(removed_nodes, add_info=add_info))

    def has_avg_distance_matrix(self, removed_nodes: [int], add_info: str = None) -> bool:
        return MemoryAccess.__has_csv(self.__get_avg_distance_matrices_name(removed_nodes, add_info=add_info))

    def remove_avg_distance_matrix(self, removed_nodes: [int], add_info: str = None) -> None:
        MemoryAccess.__remove_csv(self.__get_avg_distance_matrices_name(removed_nodes, add_info=add_info))

    def save_diff_matrix(self, removed_nodes: [int], diff: pd.DataFrame, diff_type: dt.DiffType, r_dm_index: int = None,
                         add_info: str = None) -> None:

        assert (diff_type == self.get_diff_type())
        name = self.__get_diff_matrix_name(removed_nodes, add_info=add_info)
        MemoryAccess.__save_csv(name, diff)

        if self.get_diff_type() == dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE:
            if r_dm_index is None:
                raise ValueError("r_dm_index not given")
            MemoryAccess.__save_csv(name + '_r_dm_index', pd.DataFrame([r_dm_index]))

    def load_diff_matrix(self, removed_nodes: [int], add_info: str = None):
        f_name = self.__get_diff_matrix_name(removed_nodes, add_info=add_info)
        #print("load diff matrix name:",f_name)
        diff = MemoryAccess.__load_csv(f_name)

        if self.diff_type == dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE:
            r_dm_index = MemoryAccess.__load_csv(f_name + '_r_dm_index')[0][0]
            r_dm = self.load_distance_matrix(removed_nodes=removed_nodes, iteration=r_dm_index, add_info=add_info)
        else:
            r_dm = None
        #print("loaded diff matrix type ",type(diff),type(r_dm))
        return diff, r_dm

    def has_diff_matrix(self, removed_nodes: [int], add_info: str = None) -> bool:

        name = self.__get_diff_matrix_name(removed_nodes, add_info=add_info)
        has_diff = MemoryAccess.__has_csv(name)

        if self.diff_type == dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE:
            return has_diff and MemoryAccess.__has_csv(name + "_r_dm_index")
        else:
            return has_diff


    def remove_diff_matrix(self, removed_nodes: [int], add_info: str = None) -> None:
        name = self.__get_diff_matrix_name(removed_nodes, add_info=add_info)
        MemoryAccess.__remove_csv(name)

        if self.get_diff_type() == dt.DiffType.MOST_SIMILAR_EMBS_DIFF_ONE_INIT_CONTINUE:
            MemoryAccess.__remove_csv(name + "_r_dm_index")

    def save_node_ranking_csv(self, node_ranking: pd.DataFrame, removed_nodes: [int], ranking_description: str = None):
        node_ranking.to_csv(
            self.__get_node_ranking_name(removed_nodes=removed_nodes, ranking_description=ranking_description) + ".csv")

    def load_node_ranking_csv(self, removed_nodes: [int], ranking_description: str = None):
        return pd.read_csv(
            self.__get_node_ranking_name(removed_nodes=removed_nodes, ranking_description=ranking_description) + '.csv',
            index_col=0)

    @deprecated.deprecated()
    def save_node_raking(self, removed_nodes: [int], node_ranking: list,
                         neighbours_of_removed_node: typing.List[int] = None) -> None:
        if neighbours_of_removed_node is None:
            neighbours_of_removed_node = []

        with open(self.__get_node_ranking_name(removed_nodes),
                  'w') as txt:
            for line in node_ranking:
                txt.write(f"Node: {str(line[0])} Score: {str(line[1])}" +
                          (" <--" if line[0] in neighbours_of_removed_node else "") + "\n")

    def save_training_data(self, removed_nodes: [int], feature_type: ft.FeatureType, num_of_bins: int,
                           training_data: pd.DataFrame, add_info: str = None) -> None:
        MemoryAccess.__save_csv(
            self.__get_training_data_data_name(removed_nodes=removed_nodes, feature_type=feature_type,
                                               num_of_bins=num_of_bins, add_info=add_info),
            training_data)

    def load_training_data(self, removed_nodes: [int], feature_type: ft.FeatureType, num_of_bins: int,
                           add_info: str = None) -> pd.DataFrame:
        try:
            return MemoryAccess.__load_csv(
                self.__get_training_data_data_name(removed_nodes=removed_nodes, feature_type=feature_type,
                                                   num_of_bins=num_of_bins, add_info=add_info))
        except FileNotFoundError as e:
                raise e

    def has_training_data(self, removed_nodes: [int], feature_type: ft.FeatureType, num_of_bins: int,
                          add_info: str = None) -> bool:

        return MemoryAccess.__has_csv(self.__get_training_data_data_name(removed_nodes=removed_nodes, feature_type=feature_type,
                                                   num_of_bins=num_of_bins, add_info=add_info))

    def remove_training_data(self, removed_nodes: [int], feature_type: ft.FeatureType, num_of_bins: int,
                             add_info: str = None) -> None:

        MemoryAccess.__remove_csv(
            self.__get_training_data_data_name(removed_nodes=removed_nodes, feature_type=feature_type,
                                               num_of_bins=num_of_bins, add_info=add_info))

    def load_list_of_training_data(self, removed_node: int, feature_type: ft.FeatureType, num_of_bins: int,
                                   graph: gc.Graph, tr_node_list: [int] = None,
                                   all_data_available: bool = False, limit_num: int = None) -> pd.DataFrame:
        training_data = pd.DataFrame()

        if tr_node_list is not None:
            available_graph_data = tr_node_list
            if limit_num is not None and len(available_graph_data) != limit_num:
                raise ValueError(f"The given training data does not match the number of requrired training data. \n "
                                 f"Given tr nodes {available_graph_data}, "
                                 f"should be {limit_num} but are {len(available_graph_data)}")
        elif all_data_available:
            available_graph_data = graph.nodes()
        else:
            available_graph_data = self.get_list_of_available_training_data(feature_type=feature_type,
                                                                            num_of_bins=num_of_bins, graph=graph,
                                                                            removed_first_node=removed_node)
            if limit_num is not None:
                if len(available_graph_data) < limit_num:
                    raise ValueError(f"numer of avialable graph data is smaller than the limit. \n "
                                     f"Num available graphs {available_graph_data}, limit_num {limit_num} ")
                available_graph_data = np.random.choice(available_graph_data, limit_num, replace=False)

        for other_node in available_graph_data:
            if other_node != removed_node:
                data = self.load_training_data(removed_nodes=[removed_node, other_node], feature_type=feature_type,
                                               num_of_bins=num_of_bins)
                utils.assert_df_no_nan(data, text=f"removed nodes [{removed_node}, {other_node}]")

                training_data = training_data.append(data)
                utils.assert_df_no_nan(training_data,
                                       text=f"aggregated training data after appending removed nodes"
                                            f" [{removed_node}, {other_node}]")

        utils.assert_df_no_nan(training_data, text=f"aggregated training data fro removed node {removed_node}")
        return training_data

    def load_test_data(self, removed_node: int, feature_type: ft.FeatureType, num_of_bins: int) -> pd.DataFrame:
        return self.load_training_data([removed_node], feature_type, num_of_bins=num_of_bins)

    def save_test_results(self, results: pd.DataFrame, name: str, overwrite: bool = True) -> None:
        results_name = self.__get_test_results_name(name)
        if overwrite or not os.path.exists(results_name):
            results.to_csv(results_name)
        else:
            index = 0
            while True:
                if not os.path.exists(results_name[:-4] + f"_number_{index}.csv"):
                    results.to_csv(results_name[:-4] + f"_number_{index}.csv")
                    return

    def has_test_results(self, name: str):
        return os.path.exists(self.__get_test_results_name(name=name, assure_path=False))

    def load_test_results(self, file_name: str) -> pd.DataFrame:
        file_name = self.__get_test_results_name(file_name, assure_path=False)
        return pd.read_csv(file_name, index_col=0)

    def remove_test_results(self, file_name: str) -> None:
        file_name = self.__get_test_results_name(file_name, assure_path=False)
        os.remove(file_name)

    def get_list_of_test_result_files(self) -> [str]:
        files = os.listdir(self.TEST_RESULTS_PATH)
        return files

    def access_nx_graph_edge_list_file(self, edge_list: [int], removed_nodes: [int]):

        file_name = self.__get_graph_name(removed_nodes) + ".edgelist"
        if not os.path.exists(file_name):
            # transform graph edge list into proper text format
            edges = "\n".join(list(map(lambda edge: str(edge[0]) + " " + str(edge[1]), edge_list)))

            with open(file_name, "w+") as file:
                file.write(edges)
        return file_name

    """
        insert original graph, removed_nodes are removed while writing
    """

    def access_edge_list(self, graph: gc.Graph, removed_nodes: [int] = None, graph_description: str = None):
        if removed_nodes is None:
            removed_nodes = []

        assert (all(node not in graph.nodes() for node in removed_nodes))

        file_name = self.__get_graph_name(removed_nodes, graph_description) + ".edgelist"

        if not os.path.exists(file_name):
            # create edge list file
            edges = "\n".join(list(map(lambda edge: str(edge[0]) + " " + str(edge[1]), graph.edges())))
            with open(file_name, "w+") as file:
                file.write(edges)

        return file_name

    def access_vocab(self, graph: gc.Graph, removed_nodes: [int] = None, graph_description: str = None):
        if removed_nodes is None:
            removed_nodes = []

        assert (all(node not in graph.nodes() for node in removed_nodes))
        file_name = self.__get_graph_name(removed_nodes, graph_description) + ".vocab"

        if not os.path.exists(file_name):
            # create edge list file
            nodes = "\n".join(map(lambda node: str(node) + " 0", graph.nodes()))
            with open(file_name, "w+") as file:
                file.write(nodes)

        return file_name

    def access_line_edge_list(self, graph: gc.Graph, removed_nodes: [int] = None):
        if removed_nodes is None:
            removed_nodes = None

        assert (graph.name() == self.graph)

        file_name = self.__get_graph_name(removed_nodes) + ".directedweihtededgelist"

        if not os.path.exists(file_name):
            # create edge list
            edges = "\n".join(list(map(
                lambda edge: f"{str(edge[0])} {str(edge[1])} 1\n {str(edge[1])} {str(edge[0])} 1", graph.edges())))
            with open(file_name, "w+") as file:
                file.write(edges)
        return file_name

    def get_list_of_available_embeddings(self, graph: gc.Graph, removed_first_node: int = None,
                                         emb_description: str = None,
                                         find_started_trainings: bool = False):
        files = []
        if find_started_trainings:
            iteration = 0
        else:
            iteration = self.num_iterations - 1

        if removed_first_node is not None:
            removed_nodes = [removed_first_node]
        else:
            removed_nodes = []

        for node in graph.nodes():
            if self.has_embedding(removed_nodes=removed_nodes + [node], iteration=iteration,
                                  emb_description=emb_description):
                files.append(node)

        return files

    def get_list_of_available_training_data(self, feature_type: ft.FeatureType, num_of_bins: int, graph: gc.Graph,
                                            removed_first_node: int = None):
        files = []

        if removed_first_node is not None:
            removed_nodes = [removed_first_node]
        else:
            removed_nodes = []

        for node in graph.nodes():
            if self.has_training_data(removed_nodes=removed_nodes + [node], feature_type=feature_type,
                                      num_of_bins=num_of_bins):
                files.append(node)

        assert (all([node in graph.nodes() for node in files]))

        return files

    def get_list_of_available_difference_matrices(self, graph: gc.Graph, removed_first_node: int = None):
        files = []

        if removed_first_node is not None:
            removed_nodes = [removed_first_node]
        else:
            removed_nodes = []

        for node in graph.nodes():
            if self.has_diff_matrix(removed_nodes=removed_nodes + [node]):
                files.append(node)

        return files



    def has_embeddings(self, removed_nodes, num_iterations):
        return all([self.has_embedding(removed_nodes=removed_nodes, iteration=i) for i in range(num_iterations)])

    def is_diff_type(self, diff_type):
        return self.diff_type == diff_type
