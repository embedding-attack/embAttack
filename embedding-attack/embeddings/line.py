import os
import subprocess

import memory_access as sl
import embeddings.embedding
import graphs.graph_class as gc
import config

LINE_FOLDER = config.DIR_PATH + 'LINE/linux/'


class Line(embeddings.embedding.Embedding):

    def __init__(self, dim: int = 128, threshold: int = 1000, depth: int = 2):
        assert (dim % 2 == 0)  # will be devided by 2
        self.dim: int = dim # 2 embeddings will be created and added together each emb has size dim
        self.threshold = threshold
        self.depth = depth

    def __str__(self):
        return f'LINE-dim={self.dim}_depth={self.depth}_threshold={self.threshold}__'

    def short_name(self):
        return "LINE"

    def train_embedding(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int],
                        num_of_embeddings: int, check_for_existing: bool = True):
        super().train_embedding(graph=graph, save_info=save_info, removed_nodes=removed_nodes,
                                num_of_embeddings=num_of_embeddings)

        if save_info.has_embeddings(removed_nodes=removed_nodes, num_iterations=num_of_embeddings):
            #print(f"Embeddins for removed nodes {removed_nodes} and iterations {num_of_embeddings} are already trained")
            return
        else:
            dense_edge_list = self.__get_preprocessed_edge_list(removed_nodes=removed_nodes, graph=graph,
                                                                save_info=save_info)

            for iteration in range(num_of_embeddings):
                self.__train_embedding(dense_edge_list=dense_edge_list, save_info=save_info,
                                       removed_nodes=removed_nodes,
                                       iteration=iteration, check_for_existing=check_for_existing)
            # remove unnecessary files
            os.remove(dense_edge_list)


    def load_embedding(self, graph: gc.Graph, removed_nodes: [int], save_info: sl.MemoryAccess, iteration: int,
                       load_neg_results: bool = False):

        pass

    def continue_train_embedding(self, graph: gc.Graph,
                                 save_info: sl.MemoryAccess, removed_nodes: [int],
                                 num_of_embeddings: int, model, emb_description: str = None,
                                 graph_description: str = None):
        pass


    def __get_preprocessed_edge_list(self, graph: gc.Graph, save_info: sl.MemoryAccess, removed_nodes: [int]):
        edge_list = save_info.access_edge_list(graph=graph, removed_nodes=removed_nodes)
        edgelist_name = save_info.get_graph_name(removed_nodes=removed_nodes)
        directed_weighted_edge_list = edgelist_name + ".directedWeightedEdgelist"
        dense_edge_list = edgelist_name + ".denseEdgelist"

        if os.path.exists(dense_edge_list):
            print("dense edge list already exists")
            return dense_edge_list

        if not os.path.exists(edge_list):
            raise ValueError(f"Edge list does not exist: {edge_list}")

        wd = os.getcwd()
        os.chdir(LINE_FOLDER)
        subprocess.call(f'python preprocess_youtube.py "{edge_list}" "{directed_weighted_edge_list}"', shell=True)

        if not os.path.exists(directed_weighted_edge_list):
            raise ValueError(f"Directed weighted edge list could not be computed. Target file: {edge_list}")

        subprocess.call(
            f'./reconstruct -train "{directed_weighted_edge_list}" -output "{dense_edge_list}" -depth 2 -threshold 1000',
            shell=True)
        os.chdir(wd)
        os.remove(directed_weighted_edge_list)
        assert (os.path.exists(dense_edge_list))
        return dense_edge_list

    def __train_embedding(self, dense_edge_list: [int], save_info: sl.MemoryAccess, removed_nodes: [int],
                          iteration: int, check_for_existing: bool = True):

        target_name = save_info.get_embedding_name(removed_nodes=removed_nodes, iteration=iteration)
        target_emb = target_name + ".emb"
        if check_for_existing and os.path.exists(target_emb):
            #print(f"Embedding for removed nodes {removed_nodes} and iteration {iteration} already exists.")
            return
        else:
            first_order_emb = target_name + "_order_1.emb"
            second_order_emb = target_name + "_order_2.emb"
            norm_first_order_emb = target_name + "_order_1_normalised.emb"
            norm_second_order_emb = target_name + "_order_2_normalised.emb"

            # execute embedding
            wd = os.getcwd()
            os.chdir(LINE_FOLDER)
            assert (os.path.exists(dense_edge_list))
            print("dense_edge_list", dense_edge_list)
            print("first_order_emb", first_order_emb)
            print("num cores", config.NUM_CORES)
            print("size", str(self.dim / 2))

            subprocess.call(
                f'./line -train "{dense_edge_list}" -output "{first_order_emb}" -size \
                {str(self.dim/2)} -order 1 -binary 1 -threads {config.NUM_CORES}',
                shell=True)
            subprocess.call(
                f'./line -train "{dense_edge_list}" -output "{second_order_emb}" -size \
                {str(self.dim/2)} -order 2 -binary 1 -threads {config.NUM_CORES}',
                shell=True)
            subprocess.call(f'./normalize -input "{first_order_emb}" -output "{norm_first_order_emb}" -binary 1',
                            shell=True)
            subprocess.call(f'./normalize -input "{second_order_emb}" -output "{norm_second_order_emb}" -binary 1',
                            shell=True)
            subprocess.call(
                f'./concatenate -input1 "{norm_first_order_emb}" -input2 "{norm_second_order_emb}" -output "{target_emb}" -binary 1',
                shell=True)
            os.chdir(wd)
            # remove unnecessary files to save memory
            os.remove(first_order_emb)
            os.remove(second_order_emb)
            os.remove(norm_first_order_emb)
            os.remove(norm_second_order_emb)
            assert (os.path.exists(target_emb))
    def is_static(self):
        return False

def main():
    emb = Line()
    graph = gc.Graph.init_karate_club_graph()
    save_info = sl.MemoryAccess(graph=str(graph), embedding_type=str(emb), num_iterations=1)
    emb.train_embedding(graph=graph, save_info=save_info, removed_nodes=[], num_of_embeddings=3)

    return


if __name__ == "__main__":
    main()
