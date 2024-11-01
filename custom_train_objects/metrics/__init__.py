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

import os
import keras

from keras.metrics import MeanMetricWrapper

from utils import import_objects, get_object, print_objects, is_function, dispatch_wrapper

_metrics = import_objects(
    [__package__.replace('.', os.path.sep), keras.metrics],
    classes     = keras.metrics.Metric,
    signature   = ['y_true', 'y_pred'],
    exclude     = ('Metric', 'MeanMetricWrapper')
)
globals().update(_metrics)

@dispatch_wrapper(_metrics, 'metrics')
def get_metrics(metrics, * args, ** kwargs):
    if metrics == 'accuracy': return metrics
    if isinstance(metrics, (list, tuple)):
        return [get_metrics(m, * args, ** kwargs) for m in metrics]
    
    if isinstance(metrics, dict):
        if metrics.get('class_name', None) == 'MeanMetricWrapper':
            return keras.metrics.deserialize(metrics)
        
        name_key    = 'metric' if 'metric' in metrics else 'name'
        config_key  = 'metric_config' if 'metric_config' in metrics else 'config'
        kwargs      = {** kwargs, ** metrics.get(config_key, {})}
        metric_name = metrics.get(name_key, metrics)
    
    return get_object(
        _metrics, metrics, * args, ** kwargs, types = (type, keras.metrics.Metric),
        print_name = 'metric', function_wrapper = MeanMetricWrapper
    )

def add_metric(metric, name = None):
    if name is None: name = metric.__name__ if is_function(metric) else metric.__class__.__name__
    get_metric.dispatch(metric, name)

def print_metrics():
    print_objects(_metrics, 'metrics')

