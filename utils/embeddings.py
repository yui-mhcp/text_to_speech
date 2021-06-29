import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from utils.pandas_utils import filter_df, aggregate_df
from utils.thread_utils import ThreadPool
from utils.audio.audio_io import load_audio

_allowed_embeddings_ext = ('.csv', '.npy')
_embedding_filename = 'embeddings_{}.csv'
_accepted_modes     = ('random', 'mean', 'average', 'avg', 'int', 'callable')

def get_embedding_file_ext(filename):
    for ext in _allowed_embeddings_ext:
        if os.path.exists(filename + ext): return ext
    return None

def embeddings_to_np(embeddings, col = 'embedding'):
    """ Return a numpy matrix of embeddings from a dataframe column """
    if isinstance(embeddings, str):
        if not os.path.isfile(embeddings):
            raise ValueError("You must provide an existing embedding file (got {})".format(embeddings))
        return embeddings_to_np(load_embeddings(embeddings))
    elif isinstance(embeddings, np.ndarray): return embeddings
    elif isinstance(embeddings, tf.Tensor): return embeddings.numpy()
    elif isinstance(embeddings, pd.DataFrame):
        return np.array([list(e) for e in embeddings[col].values])
    else:
        raise ValueError("Invalid type of embeddings : {}".format(type(embeddings)))

def embed_dataset(directory, dataset, embed_fn, embedding_dim, rate,
                  overwrite = False, embedding_name = _embedding_filename, 
                  max_audios = 10000, save_every = 25000, audio_kwargs = {},
                  verbose = True, tqdm = tqdm, ** kwargs):
    """
        Create embedding of all speakers in dataset and save results as csv files
        Arguments : 
            - directory : directory to save resulting csv files
            - dataset   : the pd.DataFrame dataset to embed
            - embed_fn  : embedding function, should accept a list of audios and return their embeddings
            - embedding_dim : dimension of the resulting embedding (use to create sub-directory for results)
            - rate      : rate of the audio needed by 'embed_fn'
            
            - max_audios    : number of audios to load in cache for 1 speaker
            - save_every    : number of audios to embed before saving (to avoid to restart all processing if error occurs)
            - verbose   : verbosity
            - tqdm      : progress_bar
            - embedding_name   : name of the embeddings csv file (1 embedding per utterance)
            - kwargs    : kwargs passed to 'embed_fn'
        
        Return : embeddings
            embeddings : pd.DataFrame([{'filename' :, 'embedding' :, 'id' :}, ...])
            
        
        Notes : files are saved in directory <directory>/embeddings_<embedding_dim>/
    """
    def embed_batch(batch):
        embedded = []            
        pool = ThreadPool(target = load_audio)
        if isinstance(batch, pd.DataFrame): batch = batch.to_dict('records')
        for a in batch: pool.append(kwargs = {'data' : a, 'rate' : rate, ** audio_kwargs})
        pool.start(tqdm = tqdm if verbose else lambda x: x)
        
        loaded_audios = pool.result()
        
        if verbose: kwargs['tqdm'] = tqdm
        embedded_audios = embed_fn(loaded_audios, ** kwargs)
        if isinstance(embedded_audios, tf.Tensor): embedded_audios = embedded_audios.numpy()
        
        return embedded_audios
    
    directory       = os.path.join(directory, 'embeddings')
    embeddings_file = os.path.join(directory, embedding_name.format(embedding_dim))
    
    os.makedirs(directory, exist_ok = True)
    embeddings = pd.DataFrame([], columns = ['filename', 'embedding', 'id'])
    if os.path.exists(embeddings_file) and not overwrite:
        embeddings = load_embedding(embeddings_file, with_speaker_embedding = False)
        
    processed = dataset['filename'].isin(embeddings['filename'])
    to_process = dataset[~processed].to_dict('records')
    
    if len(to_process) == 0:
        print("Dataset already processed !")
        return embeddings
        
    print("Processing dataset...\n  {} utterances already processed\n  {} utterances to process".format(len(dataset) - len(to_process), len(to_process)))
    
    n = len(embeddings)
    nb_batch = len(to_process) // max_audios + 1
    for i in range(nb_batch):
        print("Batch {} / {}".format(i+1, nb_batch))
        start_idx = i * max_audios
        batch = to_process[start_idx : start_idx + max_audios]
        if len(batch) == 0: continue

        embedded = embed_batch(batch)
        
        embedded = [
            {'filename' : infos['filename'], 'embedding' : e, 'id' : infos['id']} 
            for infos, e in zip(batch, embedded)
        ]
        embedded = pd.DataFrame(embedded)
        
        embeddings = embeddings.append(embedded)
        
        if len(embeddings) - n >= save_every:
            n = len(embeddings)
            print("Saving at utterance {}".format(len(embeddings)))
            save_embeddings(embeddings_file, embeddings)
    
    if n < len(embeddings):
        save_embeddings(embeddings_file, embeddings)
    
    return embeddings

def add_speaker_embedding(dataset, column = 'id', mode = 0):    
    if 'speaker_embedding' in dataset:
        dataset.pop('speaker_embedding')
    if column not in dataset.columns:
        dataset[column] = 'id'
    if column != 'id':
        for idx, row in dataset.iterrows():
            if column not in row or row[column] == np.nan:
                dataset.at[idx, column] = row['id']
    
    return aggregate_df(
        dataset, column, 'embedding', merge = True, speaker_embedding = mode
    )

def save_embeddings(directory, embeddings, embedding_name = _embedding_filename):
    if directory.endswith('.csv') or directory.endswith('.npy'):
        embedding_file = directory
    else:
        if not directory.endswith('embeddings'):
            directory = os.path.join(directory, 'embeddings')
        embedding_file = os.path.join(directory, embedding_name)
    
    embedding_dim = embeddings_to_np(embeddings).shape[-1]
    embedding_file = os.path.splitext(embedding_file.format(embedding_dim))[0]
    
    if isinstance(embeddings, pd.DataFrame):
        embedding_file += '.csv'
        embeddings.to_csv(embedding_file, index = False)
    elif isinstance(embeddings, (np.ndarray, tf.Tensor)):
        embedding_file += '.npy'
        embeddings = embeddings_to_np(embeddings)
        np.save(embedding_file, embeddings)
    
    return embedding_file
    
def load_embedding(directory, embedding_dim = None, dataset = None,
                   embedding_col = 'id', embedding_mode = 0, 
                   embedding_name = _embedding_filename,
                   with_speaker_embedding   = True, ** kwargs):
    """
        Load embeddings from .csv file and create an aggregation version
        
        Arguments : 
            - directory : directory on which embeddings are stored (must be 'embeddings' or have a sub-directory named 'embeddings')
            - embedding_dim : dimension of the embedding (will format embedding_name with it)
            - dataset   : the dataset on which to merge embeddings
            - embedding_name    : name for the filename
            - embedding_col / embedding_mode    : args to compute the 'speaker_embedding' col which is an aggregated version of 'embedding'
        Return :
            - embeddings or dataset merged with embeddings (merge is done on 'id' and 'filename')
    """
    embedding_name = embedding_name.format(embedding_dim)
    
    if os.path.isfile(directory):
        emb_file = directory
    else:
        if not directory.endswith('embeddings'):
            directory = os.path.join(directory, 'embeddings')
        emb_file = os.path.join(directory, embedding_name)
    
    if not os.path.exists(emb_file):
        ext = get_embedding_file_ext(emb_file)
        
        if ext is None:
            raise ValueError("Embedding file {} does not exist !".format(emb_file))
        
        emb_file += ext
    
    if emb_file.endswith('.npy'):
        return np.load(emb_file)
    
    embeddings = pd.read_csv(emb_file)
    
    embeddings['embedding'] = embeddings['embedding'].apply(
        lambda x: np.fromstring(x[1:-1], dtype = float, sep = '\t').astype(np.float32) if isinstance(x, str) else x
    )
    
    if with_speaker_embedding:
        embeddings = add_speaker_embedding(
            embeddings, embedding_col, mode = embedding_mode
        )
    
    if dataset is None: return embeddings
    
    if embeddings['id'].dtype != dataset['id'].dtype:
        embeddings['id'] = embeddings['id'].apply(lambda i: str(i))
    
    dataset = pd.merge(dataset, embeddings, on = ['id', 'filename'])

    dataset = dataset.dropna(
        axis = 'index', subset = ['embedding', 'speaker_embedding']
    )
    
    return dataset

def select_embedding(embeddings, mode = 'random', ** kwargs):
    """
        Return a single embedding (np.ndarray with rank 1) from a collection of embeddings 
        
        Arguments : 
            - embeddings    : pd.DataFrame with 'embedding' col or np.ndarray (2D matrix)
            - mode      : selection mode (int / 'avg' / 'mean' / 'average' / 'random' / callable)
        Return :
            - embedding : 1D ndarray
    """
    if isinstance(embeddings, pd.DataFrame):
        embeddings = filter_df(embeddings, ** kwargs)
        np_embeddings = embeddings_to_np(embeddings)
    else:
        np_embeddings = embeddings
    
    if isinstance(mode, int):
        return np_embeddings[mode]
    elif mode in ('mean', 'avg', 'average'):
        return np.mean(np_embeddings, axis = 0)
    elif mode == 'random':
        return np_embeddings[random.randrange(len(np_embeddings))]
    elif callable(mode):
        return mode(np_embeddings)
    else:
        raise ValueError("Mode to compute speaker embedding unknown\n  Get : {}\n  Accepted : {}".format(mode, _accepted_modes))
    
def compute_mean_embeddings(embeddings, ids):
    """ Compute the mean embeddings for each id """
    uniques = tf.unique(ids)[0]
    return uniques, tf.concat([
        tf.reduce_mean(
            tf.gather(embeddings, tf.where(ids == unique_id)), axis = 0
        )
        for unique_id in uniques
    ], axis = 0)
