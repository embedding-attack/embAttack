import sys
sys.path.insert(0, "..")

import jobs.all_exp as jobs

import graphs.graph_class as gc
import embeddings.line as line
import embeddings.node2vec_c_path_gensim_emb as n2v
import embeddings.GEM_embeddings as gem
import features.diff_type as dt

jobs.run_experiments(graphs=[gc.Graph.init_karate_club_graph()],
                     embeddings=[gem.GEM_embedding.init_hope(8), n2v.Node2VecPathSnapEmbGensim(), line.Line(),
                                 gem.GEM_embedding.init_sdne()], list_num_tr_graphs=[1], num_test_eval=1)
