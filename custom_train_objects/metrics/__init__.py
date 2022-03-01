
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

import tensorflow as tf

from custom_train_objects.metrics.metric_list import MetricList, LossMetrics

from custom_train_objects.metrics.text_accuracy import TextAccuracy
from custom_train_objects.metrics.text_metric import TextMetric
from custom_train_objects.metrics.confusion_matrix import *
from custom_train_objects.metrics.equal_error_rate import EER

_metrics = {
    'EER'                       : EER,
    'TextAccuracy'              : TextAccuracy,
    'TextMetric'                : TextMetric,
    'ConfusionMatrixMetric'     : ConfusionMatrixMetric,
    'ConfusionMatrix'           : ConfusionMatrixMetric,
    'confusion_matrix'          : ConfusionMatrixMetric,
    'TrueNegative'              : TrueNegative,
    'true_negative'             : TrueNegative,
    'TruePositive'              : TruePositive,
    'true_positive'             : TruePositive,
    
    'AUC'                       : tf.keras.metrics.AUC,
    'Acc'                       : tf.keras.metrics.Accuracy,
    'Accuracy'                  : tf.keras.metrics.Accuracy,
    'BinaryAccuracy'            : tf.keras.metrics.BinaryAccuracy,
    'binary_accuracy'           : tf.keras.metrics.BinaryAccuracy,
    'CategoricalAccuracy'       : tf.keras.metrics.CategoricalAccuracy,
    'categorical_accuracy'      : tf.keras.metrics.CategoricalAccuracy,
    'KLDivergence'              : tf.keras.metrics.KLDivergence,
    'SparseCategoricalAccuracy' : tf.keras.metrics.SparseCategoricalAccuracy,
    'sparse_categorical_accuracy'   : tf.keras.metrics.SparseCategoricalAccuracy,
}
