import multiprocessing
import os

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_PATH = os.path.abspath(CONFIG_DIR + '/../') + "/"

GEM_PATH = DIR_PATH + "/../"

NUM_CORES = multiprocessing.cpu_count()

REMOTE_DIR_PATH = "/run/user/1002/gvfs/sftp:host=alpha/home/mellers/"  # only used for evaluations

NODE2VEC_SNAP_DIR = DIR_PATH + "/snap/examples/node2vec/"