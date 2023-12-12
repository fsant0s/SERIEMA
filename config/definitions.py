import os
from dotenv import load_dotenv, find_dotenv

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
PATH_HADATASETS = "/hadatasets/fillipe.silva/"
PAHT_LOCAL_DATASETS = ROOT_DIR + "/datasets/"

VEC_SIZE = 768
RNDN = 42 #random_state
#ALGORITHMS = ['tfidf',  'doc2word', 'bert',    'roberta']
#ALGORITHMS = ['tfidf',	'doc2word',	'bert_imdb', 'roberta_imdb']
#METHODS = ['adjusted_rand_score', 'adjusted_mutual_info_score', 'bagclust', 'han', 'OTclust']

_ = load_dotenv(find_dotenv()) # This line brings all environment variables from .env into os.environ
WANDB_API = os.environ["WANDB_API_KEY"]