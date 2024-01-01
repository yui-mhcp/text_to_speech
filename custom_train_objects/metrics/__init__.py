# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import tensorflow as tf

try:
    from keras.metrics import MeanMetricWrapper
except:
    from keras.src.metrics import MeanMetricWrapper

from utils.generic_utils import import_objects, get_object, print_objects, is_function

def get_metrics(metric_name, * args, ** kwargs):
    if isinstance(metric_name, (list, tuple)):
        return [get_metrics(m, * args, ** kwargs) for m in metric_name]
    
    if isinstance(metric_name, dict):
        if 'class_name' in metric_name:
            return tf.keras.metrics.deserialize(metric_name, _metrics)
        
        name_key    = 'metric' if 'metric' in metric_name else 'name'
        config_key  = 'metric_config' if 'metric_config' in metric_name else 'config'
        kwargs      = {** kwargs, ** metric_name.get(config_key, {})}
        metric_name = metric_name.get(name_key, metric_name)
    
    return get_object(
        _metrics, metric_name, * args, ** kwargs, types = (type, tf.keras.metrics.Metric),
        err = True, print_name = 'metric', function_wrapper = MeanMetricWrapper
    )

def add_metric(metric, name = None):
    if name is None: name = metric.__name__ if is_function(metric) else metric.__class__.__name__
    
    _metrics[name] = metric

def print_metrics():
    print_objects(_metrics, 'metrics')


def _is_class_or_callable(name, val):
    return isinstance(val, type) or callable(val)

_metrics = {
    ** import_objects(__package__.replace('.', os.path.sep), types = type),
    ** import_objects(
        [tf.keras.metrics],
        filters = _is_class_or_callable,
        exclude = ('get', 'serialize', 'deserialize'),
    )
}
globals().update(_metrics)