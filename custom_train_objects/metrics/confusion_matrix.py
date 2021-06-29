import tensorflow as tf

ALL             = 0
ACCURACY        = 1
TRUE_POSITIVE   = 2
TRUE_NEGATIVE   = 3
FALSE_POSITIVE  = 4
FALSE_NEGATIVE  = 5

class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def __init__(self, true_threshold = 0.5, false_threshold = 0.5, 
                 mode = 'all', name = 'ConfusionMatrix', ** kwargs):
        super(ConfusionMatrixMetric, self).__init__(name = name, ** kwargs)
        self.true_threshold     = true_threshold
        self.false_threshold    = false_threshold
        self.mode   = self._get_mode(mode)
        
        self.samples     = self.add_weight("samples", initializer = "zeros")
        self.true_pred   = self.add_weight("true_pred", initializer = "zeros")
        self.false_pred  = self.add_weight("false_pred", initializer = "zeros")
        
        self.true_positive  = self.add_weight("true_positive", initializer = "zeros")
        self.true_negative  = self.add_weight("true_negative", initializer = "zeros")    
    
    @property
    def metric_names(self):
        if self.mode == ALL: return ['accuracy', 'true_positive', 'true_negative', 'false_positive', 'false_negative']
        elif self.mode == ACCURACY: return ['accuracy']
        elif self.mode == TRUE_POSITIVE: return ['true_positive']
        elif self.mode == TRUE_NEGATIVE: return ['true_negative']
        elif self.mode == FALSE_POSITIVE: return ['false_positive']
        elif self.mode == FALSE_NEGATIVE: return ['false_negative']
        
    def _get_mode(self, mode):
        if mode in (None, ALL, 'all', '*'): return ALL
        elif mode in (ACCURACY, 'acc', 'accuracy'): return ACCURACY
        elif mode in (TRUE_POSITIVE, 'tp', 'true_positive'): return TRUE_POSITIVE
        elif mode in (TRUE_NEGATIVE, 'tn', 'true_negative'): return TRUE_NEGATIVE
        elif mode in (FALSE_POSITIVE, 'fp', 'false_positive'): return FALSE_POSITIVE
        elif mode in (FALSE_NEGATIVE, 'fn', 'false_negative'): return FALSE_NEGATIVE
        else:
            raise ValueError("Mode inconnu : {}".format(mode))
        
    def update_state(self, y_true, y_pred):
        y_pred = tf.squeeze(y_pred, axis = 1)
        true = y_true == 1
        false = tf.logical_not(true)
        
        pos = y_pred > self.true_threshold
        neg = y_pred < self.false_threshold
        
        true_pos = tf.reduce_sum(tf.cast(
            tf.math.logical_and(true, true == pos), tf.float32
        ))
        true_neg = tf.reduce_sum(tf.cast(
            tf.math.logical_and(false, false == neg), tf.float32
        ))
                
        self.samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        self.true_pred.assign_add(tf.reduce_sum(tf.cast(pos, tf.float32)))
        self.false_pred.assign_add(tf.reduce_sum(tf.cast(neg, tf.float32)))
        
        self.true_positive.assign_add(true_pos)
        self.true_negative.assign_add(true_neg)
            
    def result(self):
        accuracy    = tf.cond(
            self.samples > 0,
            lambda: (self.true_positive + self.true_negative) / self.samples,
            lambda: 0.
        )
        true_pos    = tf.cond(
            self.true_pred > 0,
            lambda: self.true_positive / self.true_pred,
            lambda: 0.
        )
        true_neg    = tf.cond(
            self.false_pred > 0,
            lambda: self.true_negative / self.false_pred,
            lambda: 0.
        )
        false_pos    = tf.cond(
            self.true_pred > 0,
            lambda: 1. - true_pos,
            lambda: 0.
        )
        false_neg    = tf.cond(
            self.false_pred > 0,
            lambda: 1. - true_neg,
            lambda: 0.
        )
        
        if self.mode == ALL:
            return [accuracy, true_pos, true_neg, false_pos, false_neg]
        elif self.mode == ACCURACY: return accuracy
        elif self.mode == TRUE_POSITIVE: return true_pos
        elif self.mode == TRUE_NEGATIVE: return true_neg
        elif self.mode == FALSE_POSITIVE: return false_pos
        elif self.mode == FALSE_NEGATIVE: return false_neg

    def get_config(self):
        config = super(ConfusionMatrixMetric, self).get_config()
        config['true_threshold'] = self.true_threshold
        config['false_threshold'] = self.false_threshold
        config['mode'] = self.mode
        
        return config
    
class TruePositive(ConfusionMatrixMetric):
    def __init__(self, * args, name = 'true_positive', ** kwargs):
        kwargs['mode'] = 'true_positive'
        super(TruePositive, self).__init__(* args, name = name, ** kwargs)
        
        
class TrueNegative(ConfusionMatrixMetric):
    def __init__(self, * args, name = 'true_negative', ** kwargs):
        kwargs['mode'] = 'true_negative'
        super(TrueNegative, self).__init__(* args, name = name, ** kwargs)
        
        
class FalsePositive(ConfusionMatrixMetric):
    def __init__(self, * args, name = 'false_positive', ** kwargs):
        kwargs['mode'] = 'false_positive'
        super(FalsePositive, self).__init__(* args, name = name, ** kwargs)
        
        
class FalseNegative(ConfusionMatrixMetric):
    def __init__(self, * args, name = 'false_negative', ** kwargs):
        kwargs['mode'] = 'false_negative'
        super(FalseNegative, self).__init__(* args, name = name, ** kwargs)
        