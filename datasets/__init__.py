from utils.generic_utils import print_objects

from datasets.dataset_utils import *
from datasets.custom_datasets import load_dataset, _custom_datasets

_keras_datasets = {
    'cifar10'       : tf.keras.datasets.cifar10.load_data,
    'cifar100'      : tf.keras.datasets.cifar100.load_data,
    'fashion_mnist' : tf.keras.datasets.fashion_mnist.load_data,
    'imdb'          : tf.keras.datasets.imdb.load_data,
    'mnist'         : tf.keras.datasets.mnist.load_data,
    'reuters'       : tf.keras.datasets.reuters.load_data
}

def get_dataset(ds_name, ds_type = 'tf', **kwargs):
    if isinstance(ds_name, (list, tuple)): 
        if all([n in _custom_datasets for n in ds_name]):
            return load_dataset(ds_name, ** kwargs)
        else:
            return [get_dataset(n, t, **kwargs) for n, t in zip(ds_name, ds_type)]
    elif isinstance(ds_name, dict):
        return [get_dataset(n, ** ds_args) for n, ds_args in ds_name.items()]
    
    if ds_name in _custom_datasets or ds_type == 'custom':
        dataset = load_dataset(ds_name, ** kwargs)
    elif ds_type in ('tensorflow', 'tf'):
        import tensorflow_datasets as tfds
        
        dataset = tfds.load(ds_name, **kwargs)
    elif ds_type == 'keras' and ds_name in _keras_datasets:
        dataset = _keras_datasets[ds_name](**kwargs)
    else:
        raise ValueError("Dataset {} (type {}) does not exist !".format(ds_name, ds_type))
    
    return dataset

def print_datasets():
    print_objects(_keras_datasets, 'keras datasets')
    print_objects(_custom_datasets, 'custom datasets')
