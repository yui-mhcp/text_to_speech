import tensorflow as tf

class EER(tf.keras.metrics.AUC):
    def __init__(self, name = 'EER', ** kwargs):
        super(EER, self).__init__(name = name, ** kwargs)
    
    @property
    def metric_names(self):
        return ['EER', 'AUC']
                    
    def result(self):
        auc = super().result()
        
        tp_rate = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        fp_rate = tf.math.divide_no_nan(
            self.false_positives, self.false_positives + self.true_negatives
        )

        fn_rate = 1 - tp_rate
        diff = tf.abs(fp_rate - fn_rate)
        min_index = tf.math.argmin(diff)
        eer = tf.reduce_mean([fp_rate[min_index], fn_rate[min_index]])
        return eer, auc
