import tensorflow as tf

from utils import get_metric_names

class MetricList(tf.keras.metrics.Metric):
    def __init__(self, metrics, losses = None, ** kwargs):
        super().__init__(** kwargs)
        
        self._metrics   = tf.nest.flatten(metrics)
        
        if losses is not None:
            self._metrics   = [LossMetrics(losses)] + self._metrics
        
        self._names = get_metric_names(self._metrics)
    
    @property
    def metric_names(self):
        return self._names
    
    def reset_states(self):
        for metric in self._metrics: metric.reset_states()
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        for metric in self._metrics: metric.update_state(y_true, y_pred)
    
    def result(self):
        flattened = []
        for result in [m.result() for m in self._metrics]:
            result = tf.reshape(result, [-1])
            flattened.extend(tf.unstack(result))

        return flattened

class LossMetrics(tf.keras.metrics.Metric):
    def __init__(self, losses, ** kwargs):
        super().__init__(** kwargs)
        
        self._names     = get_metric_names(losses)
        self._losses    = tf.nest.flatten(losses)
        if not isinstance(self._losses, (list, tuple)):
            self._losses = [self._losses]
        if not isinstance(self._names, (list, tuple)):
            self._names = [self._names]
        
        if len(self._names) == 1: self._names = ['loss']
        self._metrics   = [LossMetric(l) for l in self._losses]
    
    @property
    def metric_names(self):
        return self._names
    
    def reset_states(self):
        for metric in self._metrics: metric.reset_states()
    
    def update_state(self, y_true, y_pred):
        for metric in self._metrics: metric.update_state(y_true, y_pred)
    
    def result(self):
        return [m.result() for m in self._metrics]

class LossMetric(tf.keras.metrics.Metric):
    def __init__(self, loss, ** kwargs):
        super().__init__(** kwargs)
        
        self._names = get_metric_names([loss])
        self._loss  = loss
        self._metrics   = [tf.keras.metrics.Mean() for _ in range(len(self._names))]
    
    @property
    def should_unstack(self):
        return len(self._names) > 1
    
    @property
    def metric_names(self):
        return self._names
    
    def reset_states(self):
        for metric in self._metrics: metric.reset_states()
    
    def update_state(self, y_true, y_pred):
        loss = self._loss(y_true, y_pred)
        
        loss = tf.unstack(loss) if self.should_unstack else [loss]

        tf.nest.map_structure(
            lambda m, v: m.update_state(v), self._metrics, loss
        )
    
    def result(self):
        return [m.result() for m in self._metrics]

