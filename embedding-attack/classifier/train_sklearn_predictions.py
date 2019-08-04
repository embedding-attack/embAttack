from functools import partial

import memory_access as sl
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import functools
import multiprocessing
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import f1_score
import pandas as pd
import utils
import typing
import config
import embeddings.node2vec_c_path_gensim_emb as n2v
import graphs.graph_class as gc
from enum import Enum
import imblearn.over_sampling as over_sampling
import imblearn.under_sampling as under_sampling

from features import diff_type as dt, feature_type as ft
from typing import List, Dict, Optional


def random_oversampling(features, labels):
    ros = over_sampling.RandomOverSampler(random_state=0)
    return ros.fit_resample(X=features, y=labels)


def random_undersampling(features, labels):
    rus = under_sampling.RandomUnderSampler(random_state=0)
    return rus.fit_resample(X=features, y=labels)


def oversampling_smote(features, labels):
    smote = over_sampling.SMOTE(random_state=0)
    return smote.fit_resample(X=features, y=labels)


def oversampling_adasyn(features, labels):
    adasyn = over_sampling.ADASYN(random_state=0)
    return adasyn.fit_resample(X=features, y=labels)


def borderline_smote(features, labels):
    smote = over_sampling.BorderlineSMOTE(random_state=0)
    return smote.fit_resample(X=features, y=labels)


def svmsmote(features, labels):
    smote = over_sampling.SVMSMOTE(random_state=0)
    return smote.fit_resample(X=features, y=labels)


class SamplingStrategy(Enum):
    RANDOM_OVERSAMPLING = partial(random_oversampling)
    RANDOM_UNDERSAMPLING = partial(random_undersampling)
    SMOTE_OVERSAMPLING = partial(oversampling_smote)
    ADASYN_OVERSAMPLING = partial(oversampling_adasyn)
    BORDERLINE_SMOTE_OVERSAMPLING = partial(borderline_smote)
    SVMSMOTE_OVERSAMPLING = partial(svmsmote)

    def sample(self, features: np.ndarray, labels: np.ndarray) -> (pd.DataFrame, pd.DataFrame):
        return self.value(features, labels)


def prepare_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray):
    features = data.drop("y", axis=1).values
    labels = data["y"].values
    labels = preprocessing.LabelEncoder().fit_transform(labels)
    return features, labels


def print_results(row_labels, prediction, labels, graph: gc.Graph):
    print("connected")
    for i in range(len(labels)):
        if labels[i] == 1:
            print(row_labels[i], "Predicted: ", prediction[i], "actual:", labels[i], "correct:",
                  prediction[i] == labels[i], "Degree of node:", graph.degree(row_labels[i]))
    print("not connected")
    for i in range(len(labels)):
        if labels[i] == 0:
            print(row_labels[i], "Predicted: ", prediction[i], "actual:", labels[i], "correct:",
                  prediction[i] == labels[i], "Degree of node:", graph.degree(row_labels[i]))


def _run_classification(training_features, training_labels, test_features, clf):
    # debug
    try:
        clf.fit(training_features, training_labels)  # fit classifier
    except Exception as e:
        for row in training_features:
            print(row)
        raise e

    train_predicted = clf.predict(training_features)
    test_predicted = clf.predict(test_features)

    train_probabilities = clf.predict_proba(training_features)[:, 1]
    test_probabilities = clf.predict_proba(test_features)[:, 1]

    return train_predicted, train_probabilities, test_predicted, test_probabilities


def _train(classification_function, train_data: pd.DataFrame, test_data: pd.DataFrame,
           sampling_strategy: typing.Optional[SamplingStrategy]):
    # load data

    train_features, train_labels = prepare_data(train_data)
    test_features, test_labels = prepare_data(test_data)

    utils.assert_df_no_nan(pd.DataFrame(train_features), text=f'Train features')
    utils.assert_df_no_nan(pd.DataFrame(train_labels), text=f'Train labels')
    utils.assert_df_no_nan(pd.DataFrame(test_features), text=f'Test features')
    utils.assert_df_no_nan(pd.DataFrame(test_labels), text=f'Test labels')

    if sampling_strategy is not None:
        train_features, train_labels = sampling_strategy.sample(train_features, train_labels)

    # train and run model
    train_predicted, train_probabilities, test_predicted, test_probabilities = \
        _run_classification(train_features, train_labels, test_features, clf=classification_function)

    return train_labels, train_predicted, train_probabilities, test_labels, test_predicted, test_probabilities


"""
def plot_degree_correct(heading: str, graph: gc.Graph, description, labels, prediction):
    print("description", description)
    print(type(list(labels == prediction)))
    print(labels == prediction)

    degrees = list(map(lambda x: graph.degree(x), description))

    pos_deg = list(compress(degrees, labels))
    pos_pred = list(compress(list(prediction == labels), labels))

    print("len res", list(prediction == labels))

    plt.title(heading)
    plt.plot(pos_deg, pos_pred, 'bo')
    # plt.plot(degrees,list(labels==prediction),'ro')
    plt.show()
    plt.close()
"""


def _auc(label, prediction):
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    return metrics.auc(fpr, tpr)


def create_ranking(labels, predicted, probabilities):
    df = pd.DataFrame(data=[labels, predicted, probabilities], index=["labels", "predicted", "probabilities"]).T
    df = df.sort_values(by="probabilities", axis=0, ascending=False)
    return df


def precision_at_k(labels, predicted, probabilities, k: int):
    ranking = create_ranking(labels, predicted, probabilities)
    ranking = ranking.head(n=k)
    return metrics.precision_score(y_true=ranking["labels"].tolist(), y_pred=ranking["predicted"].tolist(), pos_label=1)


def reciprocal_rank(labels, predicted, probabilities):
    ranking = create_ranking(labels, predicted, probabilities)
    first_pos_index = ranking["labels"].tolist().index(1)
    return 1 / (first_pos_index + 1)


def evaluate(labels, predicted, probabilities):
    results = {"error rate": 1 - metrics.accuracy_score(labels, predicted)}

    tn, fp, fn, tp = metrics.confusion_matrix(labels, predicted).ravel()
    results["true negative"] = tn
    results["false positive"] = fp
    results["false negative"] = fn
    results["true positive"] = tp

    results["accuracy"] = metrics.accuracy_score(labels, predicted)
    results["precision"] = metrics.precision_score(labels, predicted, pos_label=1)
    results["recall"] = metrics.recall_score(labels, predicted, pos_label=1)
    results["true negative rate"] = tn / (tn + fp)

    results["precision at 5"] = precision_at_k(labels, predicted, probabilities, k=5)
    results["precision at 10"] = precision_at_k(labels, predicted, probabilities, k=10)
    results["precision at 20"] = precision_at_k(labels, predicted, probabilities, k=20)

    # assert (results["accuracy"] == (tp + tn) / (tp + tn + fp + fn))

    results["auc"] = _auc(labels, probabilities)
    results["binary f1"] = f1_score(labels, predicted)

    results["reciprocal rank"] = reciprocal_rank(labels, predicted, probabilities)

    return results


def _get_avg_degree_of_neighbours(graph: gc.Graph, node: int):
    neighbours_mean = (list(map(lambda n: graph.degree(n), list(graph.neighbours(node)))))
    return sum(neighbours_mean) / len(neighbours_mean)


def _harmonic_mean(precision: int, recall: int):
    return 2 * (precision * recall) / (precision + recall)


def _create_test_results_over_all_experiments(results_per_node: pd.DataFrame):
    results_per_classifier: pd.Series = results_per_node.mean(axis=1)
    results_per_classifier.set_axis(labels=results_per_node.index, inplace=True)

    results_per_classifier["true positive"] = results_per_node.loc["true positive"].sum()
    results_per_classifier["false positive"] = results_per_node.loc["false positive"].sum()
    results_per_classifier["false negative"] = results_per_node.loc["false negative"].sum()
    results_per_classifier["true negative"] = results_per_node.loc["true negative"].sum()

    results_per_classifier.rename({"error rate": "avg error rate",
                                   "accuracy": "macro accuracy",
                                   "precision": "macro precision",
                                   "recall": "macro recall",
                                   "true negative rate": "macro true negative rate"},
                                  inplace=True)

    assert (results_per_classifier["macro precision"] == results_per_node.loc["precision"].mean())
    assert (results_per_classifier["macro recall"] == results_per_node.loc["recall"].mean())

    # compute micro values
    results_per_classifier["micro accuracy"] = (results_per_classifier["true positive"] + results_per_classifier[
        "true negative"]) / (results_per_classifier["true positive"] + results_per_classifier["true negative"] +
                             results_per_classifier["false positive"] + results_per_classifier["false negative"])
    results_per_classifier["micro precision"] = results_per_classifier["true positive"] / (
            results_per_classifier["true positive"] + results_per_classifier["false positive"])
    results_per_classifier["micro recall"] = results_per_classifier["true positive"] / (
            results_per_classifier["true positive"] + results_per_classifier["false negative"])
    results_per_classifier["micro true negative rate"] = results_per_classifier["true negative"] / (
            results_per_classifier["true negative"] + results_per_classifier["false positive"])

    # complex accuracy meaures
    auc = results_per_classifier["auc"]
    results_per_classifier.drop("auc", inplace=True)
    results_per_classifier.at["avg auc"] = auc

    results_per_classifier.drop("binary f1", inplace=True)
    results_per_classifier.at["macro f1"] = _harmonic_mean(results_per_classifier["macro precision"],
                                                           results_per_classifier["macro recall"])
    results_per_classifier.at["micro f1"] = _harmonic_mean(results_per_classifier["micro precision"],
                                                           results_per_classifier["micro recall"])

    rrank = results_per_classifier["reciprocal rank"]
    results_per_classifier.drop("reciprocal rank", inplace=True)
    results_per_classifier.at["mean rank"] = rrank

    # reoder results
    deg = results_per_classifier["degree"]
    results_per_classifier.drop("degree", inplace=True)
    results_per_classifier["avg degree"] = deg

    avg_neighbour_deg = results_per_classifier["avg_neighbour_degree"]
    results_per_classifier.drop("avg_neighbour_degree", inplace=True)
    results_per_classifier["avg_neighbour_degree"] = avg_neighbour_deg

    return results_per_classifier


def calculate_avg_distance_to_positive_predicted_nodes(graph: gc.Graph, removed_node: int, labels: [int],
                                                       predicted: [int]):
    pos_labels = labels[predicted == 1]
    if len(pos_labels) > 0:
        return float(sum(map(lambda x: graph.distance(removed_node, x), pos_labels))) / len(pos_labels)
    else:
        print(f"no node was predicted to be connected to {removed_node}")
        return 0


def replace_lable_by_2_hop_label(data: pd.DataFrame):
    pass


def get_overall_results_name(feature_type: str, sampling_strategy: typing.Optional[SamplingStrategy],
                             diff_type: dt.DiffType, num_iterations, num_tr_graphs_limit):
    assert (type(feature_type) == str)

    if sampling_strategy is not None:
        sampling_str = f"_sampling={sampling_strategy.name}"
    else:
        sampling_str = ""

    if num_tr_graphs_limit is not None:
        str_limit = f"_num_tr_graphs_{num_tr_graphs_limit}"
    else:
        str_limit = ""

    name = f"OverallTestResults_ft={feature_type}" + sampling_str + str(
        diff_type) + f"_num_iterations_{num_iterations}" + str_limit

    # if cheater_sampling:
    #    name = name + f"_cheater_sampling"

    return name


def test_per_node(nodes_to_train_on: List[int], graph: gc.Graph, save_info: sl.MemoryAccess,
                  feature_type: ft.FeatureType, num_of_bins: int, limit_num_training_graphs: Optional[int],
                  sampling_strategy: Optional,
                  c,
                  removed_node: int):
    if nodes_to_train_on is not None:
        tr_node_list = nodes_to_train_on[removed_node]
    else:
        tr_node_list = None
        raise ValueError("Training node list is not given, should be given though")
    train_data = save_info.load_list_of_training_data(removed_node=removed_node,
                                                      graph=graph.delete_node(removed_node),
                                                      feature_type=feature_type,
                                                      num_of_bins=num_of_bins,
                                                      limit_num=limit_num_training_graphs,
                                                      tr_node_list=tr_node_list)

    utils.assert_df_no_nan(train_data, text=f'Training data for removed node {removed_node}')

    test_data = save_info.load_test_data(removed_node=removed_node, feature_type=feature_type,
                                         num_of_bins=num_of_bins)
    utils.assert_df_no_nan(test_data, text=f'Test data for removed node {removed_node}')

    tr_labels, tr_predicted, tr_probabilities, te_labels, te_predicted, te_probabilities = \
        _train(c, train_data=train_data, test_data=test_data, sampling_strategy=sampling_strategy)

    # train_results, test_results = evaluate(tr_labels, tr_predicted, te_labels, te_predicted, te_probabilities)
    train_results = evaluate(tr_labels, tr_predicted, tr_probabilities)
    test_results = evaluate(te_labels, te_predicted, te_probabilities)

    # add some additional information
    test_results["degree"] = graph.degree(removed_node)

    test_results["avg_neighbour_degree"] = graph.average_neighbour_degree(removed_node)

    test_results["avg dist to pos pred"] = \
        calculate_avg_distance_to_positive_predicted_nodes(graph=graph, removed_node=removed_node,
                                                           labels=test_data.index.values,
                                                           predicted=te_predicted)

    test_results["num training features"] = len(train_data)
    test_results["num test features"] = len(test_data)

    test_results["train false negative"] = train_results["false negative"]
    test_results["train true positive"] = train_results["true positive"]
    test_results["train accuracy"] = train_results["accuracy"]
    test_results["train precision"] = train_results["precision"]
    test_results["train recall"] = train_results["recall"]
    test_results["train auc"] = train_results["auc"]

    return pd.Series(test_results), removed_node


def full_test_results_available(target_overall_file_name: str, save_info: sl.MemoryAccess, classifier: List[str]):
    # check if embedding is already trained
    if save_info.has_test_results(name=target_overall_file_name):
        overall_res: pd.DataFrame = save_info.load_test_results(file_name=target_overall_file_name)
        completed_classifier = overall_res.columns.values.tolist()
        contains_mean_rank = 'mean rank' in overall_res.index.values
        # only don't train of all classifiers are trained (for simplicity all are trained again if one is missing)
        if contains_mean_rank and set(completed_classifier) == set(map(lambda c: str(c).split("(")[0], classifier)):
            print(f"Graph {save_info.graph}, "
                  f"emb {save_info.embedding_type}, "
                  f"dt {save_info.get_diff_type().to_str()}, "
                  f"completed classifier {set(completed_classifier)}\n",
                  f"classifier to be trained {classifier}\n"
                  f"IS ALREADY TRAINED")
            return True
        else:
            print(f"Graph {save_info.graph}, "
                  f"emb {save_info.embedding_type}, "
                  f"dt {save_info.get_diff_type().to_str()}, "
                  f"completed classifier {set(completed_classifier)}\n",
                  f"classifier to be trained {classifier}\n"
                  f"IS MISSING MEAN RANK OR CLASSIFIER")
    return False


def tr_data_available(save_info: sl.MemoryAccess, graph: gc.Graph, feature_type: ft.FeatureType, num_of_bins: int,
                      list_nodes_to_predict: List[int], nodes_to_train_on: Dict[int, List[int]], classifier: [] = None,
                      sampling_strategy=None, save: bool = True,
                      limit_num_training_graphs: int = 10, check_for_existing: bool = True,
                      num_eval_iterations: int = None):
    if classifier is None:
        classifier = [KNeighborsClassifier(),
                      SVC(kernel="linear", probability=True),
                      DecisionTreeClassifier(),
                      RandomForestClassifier(),
                      AdaBoostClassifier(),
                      GaussianNB()]

    target_overall_file_name = get_overall_results_name(feature_type=feature_type.to_str(num_of_bins),
                                                        sampling_strategy=sampling_strategy,
                                                        diff_type=save_info.diff_type,
                                                        num_iterations=save_info.num_iterations,
                                                        num_tr_graphs_limit=limit_num_training_graphs)
    # check if embedding is already trained

    return full_test_results_available(target_overall_file_name=target_overall_file_name, save_info=save_info,
                                       classifier=classifier)


def test(save_info: sl.MemoryAccess, graph: gc.Graph, feature_type: ft.FeatureType, num_of_bins: int,
         list_nodes_to_predict: List[int], nodes_to_train_on: Dict[int, List[int]], classifier: [] = None,
         sampling_strategy=None, save: bool = True,
         limit_num_training_graphs: int = 10, check_for_existing: bool = True, num_eval_iterations: int = None):
    if save_info.get_diff_type().has_one_init_graph():
        if num_eval_iterations is None:
            diff_iter = range(save_info.get_num_iterations())
        else:
            diff_iter = range(num_eval_iterations)
    else:
        diff_iter = [-1]

    for i in diff_iter:
        if i >= 0:
            save_info.get_diff_type().set_iter(i)

        if classifier is None:
            classifier = [KNeighborsClassifier(),
                          SVC(kernel="linear", probability=True),
                          DecisionTreeClassifier(),
                          RandomForestClassifier(),
                          AdaBoostClassifier(),
                          GaussianNB()]  # , MLPClassifier()]

        target_overall_file_name = get_overall_results_name(feature_type=feature_type.to_str(num_of_bins),
                                                            sampling_strategy=sampling_strategy,
                                                            diff_type=save_info.diff_type,
                                                            num_iterations=save_info.num_iterations,
                                                            num_tr_graphs_limit=limit_num_training_graphs)
        # check if embedding is already trained

        if check_for_existing and full_test_results_available(target_overall_file_name=target_overall_file_name,
                                                              save_info=save_info, classifier=classifier):
            continue
        """
        if list_nodes_to_predict is None:
            raise ValueError("Safty Error: the list of nodes to predict is not given.")
            list_nodes_to_predict = save_info.get_list_of_available_training_data(graph=graph,
                                                                                  feature_type=feature_type,
                                                                                  num_of_bins=num_of_bins)
        """
        assert (len(list_nodes_to_predict) > 0)
        print(f"data is available for nodes: {list_nodes_to_predict}")

        overall_results = pd.DataFrame()

        for c in classifier:
            results_per_node = pd.DataFrame()

            # train_labels = []
            # train_predicted = []
            # train_probabilities = []
            # test_labels = []
            # test_predicted = []
            # test_probabilities = []

            exp_per_node = functools.partial(test_per_node, nodes_to_train_on, graph,
                                             save_info, feature_type,
                                             num_of_bins,
                                             limit_num_training_graphs,
                                             sampling_strategy, c)

            with multiprocessing.Pool(min(config.NUM_CORES, len(list_nodes_to_predict))) as pool:
                for res in pool.imap(exp_per_node, list_nodes_to_predict):
                    results_per_node[res[1]] = res[0]

            if sampling_strategy:
                sampling_str = f"_sampling={sampling_strategy.to_str(num_of_bins)}"
            else:
                sampling_str = ""
            if limit_num_training_graphs is not None:
                str_limit = f"_num_tr_graphs_{limit_num_training_graphs}"
            else:
                str_limit = ""
            if save:
                save_info.save_test_results(results_per_node.T,
                                            f"TestResults_ft={feature_type.to_str(num_of_bins)}_Cassifier=" +
                                            str(save_info.diff_type) +
                                            str(c).split("(")[0] + sampling_str +
                                            f"_num_iterations_{save_info.num_iterations}" +
                                            str_limit)

            results_per_classifier = _create_test_results_over_all_experiments(results_per_node)

            overall_results[str(c).split("(")[0]] = pd.Series(results_per_classifier.T)

        if save:
            save_info.save_test_results(results=overall_results, name=target_overall_file_name)

        print(f"Graph {save_info.graph}, "
              f"emb {save_info.embedding_type}, "
              f"dt {save_info.get_diff_type().to_str()}, "
              f"ft {feature_type.to_str(num_of_bins)}, "
              f"limit_tr_graphs {limit_num_training_graphs}")
        print(overall_results)


def test_all_sampling_strats(save_info: sl.MemoryAccess, graph: gc.Graph, feature_type: ft.FeatureType,
                             num_of_bins: int):
    # test(save_info=save_info, graph=graph, feature_type=feature_type, num_of_bins=num_of_bins)
    for strat in SamplingStrategy:
        test(save_info=save_info,
             graph=graph,
             feature_type=feature_type, num_of_bins=num_of_bins, list_nodes_to_predict=graph.nodes(),
             sampling_strategy=strat)


def main():
    emb = n2v.Node2VecPathSnapEmbGensim()
    graphs = [gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()]

    feature_type = ft.FeatureType.DIFF_BIN_WITH_DIM
    diff_type = dt.DiffType.DIFFERENCE
    num_bins = 10
    for graph in graphs:
        # base_dir = "/run/user/1002/gvfs/sftp:host=alpha/home/mellers/"
        save_info = sl.MemoryAccess(graph=str(graph), embedding_type=str(emb), num_iterations=30,
                                    diff_type=diff_type,
                                    use_remote_data=False)

        classifier = [  # KNeighborsClassifier(),
            # SVC(kernel="linear", probability=True),
            # DecisionTreeClassifier(),
            # RandomForestClassifier(),
            # AdaBoostClassifier(),
            # GaussianNB(),
            MLPClassifier(max_iter=100000, hidden_layer_sizes=(100, 100,))]

        test(save_info=save_info, graph=graph, feature_type=feature_type, num_of_bins=num_bins,
             classifier=classifier)


if __name__ == '__main__':
    main()
