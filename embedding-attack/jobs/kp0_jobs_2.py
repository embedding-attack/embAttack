import sys

sys.path.insert(0, "..")

import logging
import config

import jobs.all_exp as jobs

import graphs.graph_class as gc
import embeddings.line as line
import embeddings.node2vec_c_path_gensim_emb as n2v
import embeddings.GEM_embeddings as gem
import features.diff_type as dt

formatter = logging.Formatter('%(asctime)s %(message)s')

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

fh = logging.FileHandler(config.DIR_PATH + f'{__name__}.log')
fh.setFormatter(formatter)
logger.addHandler(fh)



logger.info('kp0 exp 1: all graphs')
jobs.run_experiments(
    graphs=gc.Graph.init_all_but_barabasi(),
    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line(), gem.GEM_embedding.init_hope()],
    logger=logger, num_eval_iter=5, assure_used_nodes=True, all_embs_trained=True)

# exp num iterations
logger.info('kp1 exp 4: exp on diff type difference')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line(), gem.GEM_embedding.init_hope()],
    list_num_iter=[1, 2, 5, 10], diff_types=[dt.DiffType.DIFFERENCE],
    logger=logger, all_embs_trained=True, assure_used_nodes=True, num_eval_iter=5)

# exp num bins
logger.info('kp1 exp 2: different number of bins')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[line.Line(), n2v.Node2VecPathSnapEmbGensim(), gem.GEM_embedding.init_hope()],
    list_num_bins=[5, 10, 20, 50, 100, 200, 500],
    logger=logger, all_embs_trained=True, assure_used_nodes=True, num_eval_iter=5)

# exp num used training graphs
logger.info('kp1 exp 3: different number of training graphs')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[line.Line(), n2v.Node2VecPathSnapEmbGensim(), gem.GEM_embedding.init_hope()],
    list_num_tr_graphs=[1, 2, 5, 10],
    logger=logger, all_embs_trained=True, assure_used_nodes=True, num_eval_iter=5)

# exp 3 dimensions
logger.info('kp1 exp 0: different dimensionality of embeddings')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[n2v.Node2VecPathSnapEmbGensim(dim=64), n2v.Node2VecPathSnapEmbGensim(dim=128),
                n2v.Node2VecPathSnapEmbGensim(dim=256), n2v.Node2VecPathSnapEmbGensim(dim=512),
                line.Line(dim=64), line.Line(dim=128), line.Line(dim=256), line.Line(dim=512)],
    logger=logger, num_eval_iter=5, all_embs_trained=False, assure_used_nodes=True)

# exp 3 dimensions
logger.info('kp1 exp 6: hope different dimensionality of embeddings')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[gem.GEM_embedding.init_hope(dim=64), gem.GEM_embedding.init_hope(dim=256),
                gem.GEM_embedding.init_hope(dim=512)],
    logger=logger, num_eval_iter=5, assure_used_nodes=False, all_embs_trained=False)
