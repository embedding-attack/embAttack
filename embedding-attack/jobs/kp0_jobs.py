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

fh = logging.FileHandler(config.DIR_PATH + 'kp0_jobs.log')
fh.setFormatter(formatter)
logger.addHandler(fh)

# exp num bins
logger.info('kp0 exp 1: different number of bins')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[gem.GEM_embedding.init_hope(),gem.GEM_embedding.init_sdne()], list_num_bins=[5, 10, 20, 50, 100, 200, 500],
    logger=logger)

# exp num used training graphs
logger.info('kp0 exp 2: different number of training graphs')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[gem.GEM_embedding.init_hope(),gem.GEM_embedding.init_sdne()], list_num_tr_graphs=[1, 2, 5, 10], logger=logger)



# exp num iterations
logger.info('kp0 exp 3: different num iterations')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line(), gem.GEM_embedding.init_hope(),
                gem.GEM_embedding.init_sdne()], list_num_iter=[1, 2, 5, 10], logger=logger)

logger.info('kp0 exp 4: all graphs')
jobs.run_experiments(
    graphs=gc.Graph.init_all_but_barabasi(),
    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line(), gem.GEM_embedding.init_hope(),
                gem.GEM_embedding.init_sdne()], logger=logger)


logger.info('kp0 exp 5: all diffs')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line()], diff_types=dt.DiffType.get_list_of_all_diff_types(),
    logger=logger)

# logger.info('kp0 exp 3: all embeddings')
# jobs.run_experiments(
#    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
#    embeddings=[n2v.Node2VecPathSnapEmbGensim(), line.Line()] + list(gem.GEM_embedding.init_list_of_gem_embeddings()),
#    logger=logger)

# exp 3 dimensions

logger.info('kp0 exp 6: different dimensionality of embeddings')
jobs.run_experiments(
    graphs=[gc.Graph.init_facebook_wosn_2009_snowball_sampled_2000()],
    embeddings=[n2v.Node2VecPathSnapEmbGensim(dim=64), n2v.Node2VecPathSnapEmbGensim(dim=128),
                n2v.Node2VecPathSnapEmbGensim(dim=256), n2v.Node2VecPathSnapEmbGensim(dim=512),
                line.Line(dim=64), line.Line(dim=128), line.Line(dim=256), line.Line(dim=512)], logger=logger)


