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
