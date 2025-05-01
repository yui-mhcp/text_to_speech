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

from .loss_with_multiple_outputs import LossWithMultipleOutputs

@keras.saving.register_keras_serializable('tacotron2')
class TacotronLoss(LossWithMultipleOutputs):
    def __init__(self,
                 from_logits        = False, 
                 label_smoothing    = 0,
                 
                 mel_loss   = 'mse',
                 mask_mel_padding   = True,
                 similarity_function    = None,
                 
                 finish_weight      = 1.,
                 not_finish_weight  = 1.,
                 
                 name   = 'tacotron_loss',
                 ** kwargs
                ):
        """
            Loss for the Tacotron-2 model architecture
            
            Arguments : 
                - from_logits / label_smooting  : for the binary_crossentropy loss computed on gates outputs
                - mask_mel_padding  : whether to mask padding or not (in mel losses)
                
                - finish_weight     : weight of binary_crossentropy loss for 1-gate
                - not_finish_weight : weight of binary_crossentropy loss for 0-gate
        """
        super().__init__(name = name, ** kwargs)
        self.mask_mel_padding   = mask_mel_padding
        self.label_smoothing    = label_smoothing
        self.from_logits        = from_logits
        self.finish_weight      = finish_weight
        self.not_finish_weight  = not_finish_weight
        self.mel_loss   = mel_loss
        self.similarity_function    = similarity_function
    
    @property
    def output_names(self):
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
        
        if loss == 'similarity':
            if mask is not None: y_pred *= mask
            similarity = self.similarity_function(y_true, y_pred)
            return K.reshape(K.binary_crossentropy(
                K.ones_like(similarity), similarity
            ), [-1])

        if isinstance(loss, str):
            if 'mse' in loss:
                error   = K.square(y_true - y_pred)
            elif 'mae' in loss:
                error   = K.abs(y_true - y_pred)
            else:
                raise ValueError("Unknown loss : {}".format(loss))
            
            if 'weighted' in loss:
                weights = y_true - K.min(y_true, keepdims = True, axis = [1, 2]) + 1.
                weights = weights / K.max(weights, keepdims = True, axis = [1, 2])
                error   = error * weights
        elif callable(loss):
            error   = loss(y_true, y_pred)
        else:
            raise ValueError("Unknown loss : {}".format(loss))
        
        if len(K.shape(error)) == 3: error = K.sum(error, axis = 2)
        
        if mask is None:
            return K.divide_no_nan(
                K.sum(error, axis = 1),
                K.cast(K.shape(y_pred)[1] * K.shape(y_pred)[2], error.dtype)
            )
        
        return K.divide_no_nan(
            K.sum(error * mask, axis = 1),
            K.sum(mask, axis = 1) * K.cast(K.shape(y_pred)[2], error.dtype)
        )
    
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
        
        gate_mask   = gate_target * self.finish_weight + (
            (1. - gate_target) * self.not_finish_weight
        )
        gate_loss = K.binary_crossentropy(
            gate_target, gate_pred,  from_logits = self.from_logits,
        )
        gate_loss = K.mean(gate_loss * gate_mask, axis = 1)

        ####################
        # Compute mel loss #
        ####################
        
        mask = K.cast(1. - gate_target, mel_pred.dtype) if self.mask_mel_padding else None
        
        mel_loss            = self.compute_mel_loss(mel_target, mel_pred, mask = mask)
        mel_postnet_loss    = self.compute_mel_loss(mel_target, mel_postnet_pred, mask = mask)
        
        if not isinstance(mel_loss, list): mel_loss = [mel_loss]
        if not isinstance(mel_postnet_loss, list): mel_postnet_loss = [mel_postnet_loss]
        
        ##############################
        #     Compute final loss     #
        ##############################
        
        loss = gate_loss
        loss += K.sum(K.stack(mel_loss, axis = 0), axis = 0) if len(mel_loss) > 1 else mel_loss[0]
        loss += K.sum(K.stack(mel_postnet_loss, axis = 0), axis = 0) if len(mel_postnet_loss) > 1 else mel_postnet_loss[0]

        return [loss] + mel_loss + mel_postnet_loss + [gate_loss]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'mask_mel_padding'  : self.mask_mel_padding,
            'label_smoothing'   : self.label_smoothing,
            'finish_weight'     : self.finish_weight,
            'not_finish_weight' : self.not_finish_weight,
            'from_logits'   : self.from_logits,
            'mel_loss'  : self.mel_loss if not callable(self.mel_loss) else keras.losses.serialize(self.mel_loss)
        })
        return config

