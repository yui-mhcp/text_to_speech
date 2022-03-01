
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
