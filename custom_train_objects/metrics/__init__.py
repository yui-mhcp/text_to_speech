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

import keras

from keras.metrics import MeanMetricWrapper

from utils.generic_utils import import_submodules

_metrics = {}
for module in [keras.metrics] + import_submodules(__package__):
    _metrics.update({
        k : v for k, v in vars(module).items()
        if (not k.startswith('_')) and (
            (isinstance(v, type) and issubclass(v, keras.metrics.Metric)) or (callable(v))
        )
    })
globals().update(_metrics)

_metrics = {k.lower() : v for k, v in _metrics.items()}

def get_metrics(metrics, ** kwargs):
    if metrics == 'accuracy': return metrics
    if isinstance(metrics, (list, tuple)):
        return [get_metrics(m, ** kwargs) for m in metrics]
    
    if isinstance(metrics, dict):
        if metrics.get('class_name', None) == 'MeanMetricWrapper':
            return keras.metrics.deserialize(metrics)
        
        name_key    = 'class_name' if 'class_name' in metrics else 'name'
        config_key  = 'metric_config' if 'metric_config' in metrics else 'config'
        kwargs      = {** kwargs, ** metrics.get(config_key, {})}
        metrics     = metrics.get(name_key, metrics)
    
    if isinstance(metrics, str):
        metrics = _metrics[metrics.lower()]
        if isinstance(metrics, type): metrics = metrics(** kwargs)
    
    if not isinstance(metrics, keras.metrics.Metric):
        assert callable(metrics), str(metrics)
        kwargs.setdefault('name', metrics.__name__)
        metrics = MeanMetricWrapper(metrics, ** kwargs)
    
    return metrics