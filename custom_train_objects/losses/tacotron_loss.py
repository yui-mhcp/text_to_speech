
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

class TacotronLoss(tf.keras.losses.Loss):
    def __init__(self,
                 from_logits        = False, 
                 label_smoothing    = 0,
                 
                 mel_loss   = 'mse',
                 mask_mel_padding   = True,
                 similarity_function    = None,
                 
                 finish_weight      = 1.,
                 not_finish_weight  = 1.,
                 
                 reduction  = 'none', 
                 name   = 'tacotron_loss',
                 **kwargs
                ):
        """
            Loss for the Tacotron-2 model architecture
            
            Arguments : 
                - from_logits / label_smooting  : for the binary_crossentropy loss computed on gates outputs
                - mask_mel_padding  : whether to mask padding or not (in mel losses)
                
                - finish_weight     : weight of binary_crossentropy loss for 1-gate
                - not_finish_weight : weight of binary_crossentropy loss for 0-gate
        """
        super().__init__(name = name, reduction = 'none', **kwargs)
        self.mask_mel_padding   = mask_mel_padding
        self.label_smoothing    = label_smoothing
        self.from_logits        = from_logits
        self.finish_weight      = finish_weight
        self.not_finish_weight  = not_finish_weight
        self.mel_loss   = mel_loss
        self.similarity_function    = similarity_function
        
    @property
    def loss_names(self):
        loss_fn_names = self.mel_loss
        if not isinstance(loss_fn_names, (list, tuple)): loss_fn_names = [loss_fn_names]

        names = ['loss']
        names += ['{}_mel_loss'.format(l) for l in loss_fn_names]
        names += ['{}_mel_postnet_loss'.format(l) for l in loss_fn_names]
        return names + ['gate_loss']
    
    def compute_mel_loss(self, y_true, y_pred, loss = None, mask = None):
        """
            Compute mel loss for given mel ground truth / pred wiith for a given loss function
            Arguments : 
                - y_true / y_pred : mel ground truth / predicted with shape [B, seq_len, n_channels]
                - loss : the loss function name (mae, mse or similarity supported)
                - mask : binary mask of shape [B, seq_len], frames to mask
            Return :
                - mel_loss per sample (shape = [B])
            
            Note : the loss is computed (for mae / mse) as the mean over the entire spectrogram (without masking if mask is provided) and **not** the mean over frames.
        """
        if loss is None: loss = self.mel_loss
        if isinstance(loss, (list, tuple)):
            return [self.compute_mel_loss(y_true, y_pred, l, mask = mask) for l in loss]
        
        if mask is not None: mask = tf.cast(mask, y_pred.dtype)
        
        if loss == 'similarity':
            if mask is not None: y_pred *= mask
            similarity = self.similarity_function(tf.stop_gradient(y_true), y_pred)
            return tf.reshape(tf.keras.losses.binary_crossentropy(
                tf.ones_like(similarity), similarity
            ), [-1])
        elif callable(loss):
            error   = loss(y_true, y_pred)
        elif loss == 'mse':
            error   = tf.square(y_true - y_pred)
        elif loss == 'mae':
            error   = tf.abs(y_true - y_pred)
        else:
            raise ValueError("Unknown loss : {}".format(loss))
        
        if len(tf.shape(error)) == 3: error = tf.reduce_sum(error, axis = -1)
        
        n_channels = tf.shape(y_pred)[-1]
        if mask is None:
            return tf.reduce_sum(error, axis = -1) / tf.cast(tf.shape(y_pred)[1] * n_channels, error.dtype)
        
        return tf.reduce_sum(error * mask, axis = -1) / (tf.reduce_sum(mask, axis = -1) * tf.cast(n_channels, mask.dtype))
    
    def call(self, y_true, y_pred):
        """
            Compute the Tacotron-2 loss as follows : 
                loss = loss_mel + loss_mel_postnet + loss_gate
                - loss_mel          = `self.mel_loss` on mel-spectrogram (before postnet)
                - loss_mel_postnet  = `self.mel_loss` on mel-spectrogram (after postnet)
                - loss_gate         = `binary_crossentropy` on gates
            
            Arguments : 
                - y_true : [mel_target, gate_target], expected spectrogram / gate
                - y_pred : [mel, mel_postnet, gates, ...], prediction of Tacotron2
            
            Shapes :
                - mel_target, mel, mel_postnet  : [batch_size, seq_len, n_channels]
                - gate_target, gates    : [batch_size, seq_len]
                - output    : [4, batch_size] (4 is for [loss, mel_loss, mel_postnet_loss, gate_loss])
        """
        mel_target, gate_target = y_true
        mel_pred, mel_postnet_pred, gate_pred = y_pred[:3]
        
        ##############################
        #      Compute gate loss     #
        ##############################
        
        reshaped_gate_target = tf.reshape(gate_target, [-1, 1])
        reshaped_gate_pred   = tf.reshape(gate_pred, [-1, 1])
        
        finish_mask     = reshaped_gate_target * self.finish_weight
        not_finish_mask = (1 - reshaped_gate_target) * self.not_finish_weight
        gate_loss = tf.keras.losses.binary_crossentropy(
            reshaped_gate_target, reshaped_gate_pred, 
            from_logits = self.from_logits, 
            label_smoothing = self.label_smoothing
        )
        gate_loss = gate_loss * finish_mask + gate_loss * not_finish_mask
        gate_loss = tf.reduce_mean(tf.reshape(gate_loss, [tf.shape(mel_target)[0], -1]), axis = -1)

        ####################
        # Compute mel loss #
        ####################
        
        mask = 1. - gate_target if self.mask_mel_padding else None
        
        mel_loss            = self.compute_mel_loss(mel_target, mel_pred, mask = mask)
        mel_postnet_loss    = self.compute_mel_loss(mel_target, mel_postnet_pred, mask = mask)
        
        if not isinstance(mel_loss, list): mel_loss = [mel_loss]
        if not isinstance(mel_postnet_loss, list): mel_postnet_loss = [mel_postnet_loss]
        
        ##############################
        #     Compute final loss     #
        ##############################
        
        loss  = tf.reduce_sum(mel_loss, 0) + tf.reduce_sum(mel_postnet_loss, 0) + gate_loss

        return tf.stack([loss] + mel_loss + mel_postnet_loss + [gate_loss], 0)
    
    def get_config(self):
        config = super().get_config()
        config['mel_loss']      = self.mel_loss
        config['mask_mel_padding']  = self.mask_mel_padding
        config['label_smoothing']   = self.label_smoothing
        config['finish_weight']     = self.finish_weight
        config['not_finish_weight'] = self.not_finish_weight
        config['from_logits']   = self.from_logits
        return config

