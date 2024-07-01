# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras

from loggers import timer
from utils.generic_utils import import_objects, print_objects

from .builder import *
from .summary import *
from .custom_datasets import set_dataset_dir, get_dataset_dir, is_custom_dataset, load_custom_dataset, show_custom_datasets

@timer(name = 'dataset loading')
def get_dataset(dataset, *, source = 'keras', ** kwargs):
    if isinstance(dataset, (list, tuple)):
        if all(is_custom_dataset(ds) for ds in dataset):
            return load_custom_dataset(dataset, ** kwargs)
        return [get_dataset(ds, source = source, ** kwargs) for ds in dataset]
    
    elif isinstance(dataset, dict):
        return [get_dataset(ds, ** config) for ds, config in dataset.items()]
    
    if is_custom_dataset(dataset) or source == 'custom':
        dataset = load_custom_dataset(dataset, ** kwargs)
    elif source in ('tensorflow', 'tensorflow_datasets', 'tf', 'tfds'):
        import tensorflow_datasets as tfds
        
        dataset = tfds.load(dataset, ** kwargs)
    elif source == 'keras' and dataset in _keras_datasets:
        dataset = _keras_datasets[dataset](** kwargs)
    elif callable(source):
        dataset = source(dataset, ** kwargs)
    else:
        raise ValueError("Dataset {} and source {} are not supported !".format(dataset, source))
    
    return dataset

def print_datasets():
    show_custom_datasets()
    print_objects(_keras_datasets, 'keras datasets')

_keras_datasets = {
    k : getattr(v, 'load_data')
    for k, v in import_objects(keras.datasets, allow_modules = True).items()
    if hasattr(v, 'load_data')
}
