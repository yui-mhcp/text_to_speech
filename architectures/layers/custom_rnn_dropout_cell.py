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
import keras.ops as K

from keras import tree

class CustomRNNDropoutCell:
    def init_dropout_mask(self, 
                          inputs    = None,
                          state     = None,
                          
                          batch_size    = None,
                          seq_length    = None,
                          
                          training  = None
                         ):
        if not training or keras.backend.backend() == 'tensorflow': return
        
        if batch_size is None or seq_length is None:
            batch_size, seq_length = K.shape(inputs)[0], K.shape(inputs)[1]
        
        if not hasattr(self, '_dropout_mask'):
            self._dropout_mask = None
        if self._dropout_mask is None and self.drop_rate > 0.:
            if isinstance(self.dropout_shape, dict):
                mask_shapes = {
                    name : (batch_size, seq_length) + shape
                    for name, shape in self.dropout_shape.items()
                }
                self._dropout_mask = {
                    name : keras.random.dropout(
                        K.ones(shape), self.drop_rate, seed = self.seed_generator
                    )
                    for name, shape in mask_shapes.items()
                }
            else:
                input_shape = (batch_size, seq_length) + self.dropout_shape
                self._dropout_mask = keras.random.dropout(
                    K.ones(input_shape), self.drop_rate, seed = self.seed_generator
                )
        
        if hasattr(self, 'rnn_cells'):
            if not tree.is_nested(state): state = [state]
            for (cell, inp_shape), s in zip(self.rnn_cells, state):
                cell.get_dropout_mask(K.zeros((batch_size, seq_length) + inp_shape))
                cell.get_recurrent_dropout_mask(s)
            

    def reset_dropout_mask(self):
        if keras.backend.backend() == 'tensorflow': return
        
        self._dropout_mask = None
        for layer in self._layers:
            if hasattr(layer, 'get_dropout_mask'):
                layer.reset_dropout_mask()
                layer.reset_recurrent_dropout_mask()
            elif hasattr(layer, 'init_dropout_mask'):
                layer.reset_dropout_mask(inputs, training = training)
            elif isinstance(layer, keras.layers.StackedRNNCells):
                for l in layer.cells:
                    l.reset_dropout_mask()
                    l.reset_recurrent_dropout_mask()

    def dropout(self, inputs, step, training, name = None):
        if not training: return inputs
        elif keras.backend.backend() == 'tensorflow':
            return keras.random.dropout(inputs, self.drop_rate, seed = self.seed_generator)
        
        if len(K.shape(step)) == 2: step = step[0, 0]
        if getattr(self, '_dropout_mask', None) is None:
            raise RuntimeError('You must call `self.init_dropout_mask` before the loop')
        
        if isinstance(self._dropout_mask, dict):
            if name is None:
                raise RuntimeError('You must provide the `name` to get the right dropout mask')
            if name not in self._dropout_mask:
                raise ValueError('Unknown dropout mask {} - availables : {}'.format(
                    name, tuple(self._dropout_mask.keys())
                ))
            
            return inputs * self._dropout_mask[name][:, step]
        return inputs * self._dropout_mask[:, step]
        
        
