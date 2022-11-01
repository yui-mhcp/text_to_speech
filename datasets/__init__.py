
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from loggers import timer
from utils.generic_utils import print_objects

from datasets.dataset_utils import *
from datasets.custom_datasets import set_dataset_dir, get_dataset_dir, show_datasets, load_dataset, _custom_datasets
from datasets.sqlite_dataset import SQLiteDataset

_keras_datasets = {
    'cifar10'       : tf.keras.datasets.cifar10.load_data,
    'cifar100'      : tf.keras.datasets.cifar100.load_data,
    'fashion_mnist' : tf.keras.datasets.fashion_mnist.load_data,
    'imdb'          : tf.keras.datasets.imdb.load_data,
    'mnist'         : tf.keras.datasets.mnist.load_data,
    'reuters'       : tf.keras.datasets.reuters.load_data
}

@timer(name = 'dataset loading')
def get_dataset(ds_name, ds_type = 'tf', ** kwargs):
    if isinstance(ds_name, (list, tuple)): 
        if all([n in _custom_datasets for n in ds_name]):
            return load_dataset(ds_name, ** kwargs)
        else:
            return [get_dataset(n, t, ** kwargs) for n, t in zip(ds_name, ds_type)]
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
    show_datasets()
    print_objects(_keras_datasets, 'keras datasets')
