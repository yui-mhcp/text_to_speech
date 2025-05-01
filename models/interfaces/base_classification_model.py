# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from functools import cached_property

from utils.keras import ops
from .base_model import BaseModel

class BaseClassificationModel(BaseModel):
    def _init_labels(self, labels, nb_class = None, ** kwargs):
        if not isinstance(labels, (list, tuple)): labels = [labels]
        
        self.labels = [str(label) for label in labels]
        self.nb_class   = max(len(self.labels), nb_class if nb_class is not None else 1)
        if self.nb_class > len(self.labels):
            self.labels += [''] * (self.nb_class - len(self.labels))
        
        self.label_to_idx   = {label : i for i, label in enumerate(self.labels)}
    
    def _str_labels(self):
        return '- Labels (n = {}) : {}\n'.format(
            self.nb_class,
            self.labels if len(self.labels) <= 10 else '[{}, ...]'.format(
                str(self.labels[:10])[1:-1]
            )
        )

    @cached_property
    def lookup_table(self):
        import tensorflow as tf
        
        keys, values = list(zip(* self.label_to_idx.items()))
        
        init  = tf.lookup.KeyValueTensorInitializer(
            tf.as_string(keys), tf.cast(values, tf.int32)
        )
        return tf.lookup.StaticHashTable(init, default_value = -1)
    
    def get_label_id(self, data):
        if isinstance(data, dict): data = data['label']
        
        if isinstance(data, (list, tuple)) or isinstance(data, np.ndarray):
            return [self.label_to_idx.get(str(label), -1) for label in data]
        elif isinstance(data, (int, str)):
            return self.label_to_idx.get(str(data), -1)
        elif ops.is_tensorflow_graph():
            import tensorflow as tf
            return self.lookup_table.lookup(tf.as_string(data))
        else:
            raise ValueError('Unsupported label (type {}) : {}'.format(type(data), data))
    
    def get_config_labels(self):
        return {'labels' : self.labels, 'nb_class' : self.nb_class}

    