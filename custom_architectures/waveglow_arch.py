
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

# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import logging
import tensorflow as tf

from custom_layers import Invertible1x1Conv

def add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = tf.tanh(in_act[:, :, :n_channels])
    s_act = tf.sigmoid(in_act[:, :, n_channels:])
    acts = t_act * s_act
    return acts

class WaveglowBlock(tf.keras.Model):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self,
                 n_in_channels,
                 n_mel_channels,
                 n_layers       = 8,
                 n_channels     = 256,
                 kernel_size    = 3,
                 ** kwargs):
        super().__init__(** kwargs)
        
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        
        self.n_in_channels  = n_in_channels
        self.n_mel_channels = n_mel_channels
        self.n_layers       = n_layers
        self.n_channels     = n_channels
        self.kernel_size    = kernel_size
        
        self.in_layers          = []
        self.res_skip_layers    = []
        self.cond_layers        = []

        self.start = tf.keras.layers.Conv1D(n_channels, 1)

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.end = tf.keras.layers.Conv1D(
            2 * n_in_channels, 1, kernel_initializer = 'zeros'
        )

        for i in range(n_layers):
            dilation = 2 ** i

            self.in_layers.append(tf.keras.layers.Conv1D(
                2 * n_channels, kernel_size, dilation_rate = dilation, padding = 'same'
            ))

            self.cond_layers.append(tf.keras.layers.Conv1D(
                2 * n_channels, 1
            ))

            # last one is not necessary
            res_skip_channels = 2 * n_channels if i < n_layers - 1 else n_channels

            self.res_skip_layers.append(tf.keras.layers.Conv1D(
                res_skip_channels, 1
            ))

    def call(self, forward_input, training = False, mask = None):
        audio, spect = forward_input
        
        audio = self.start(audio)

        for i in range(self.n_layers):
            acts = add_tanh_sigmoid_multiply(
                self.in_layers[i](audio),
                self.cond_layers[i](spect),
                self.n_channels
            )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:,:,:self.n_channels] + audio
                skip_acts = res_skip_acts[:,:,self.n_channels:]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        
        return self.end(output)

    def get_config(self):
        config = {
            'n_in_channels'     : self.n_in_channels,
            'n_mel_channels'    : self.n_mel_channels,
        
            'n_layers'       : self.n_layers,
            'n_channels'     : self.n_channels,
            'kernel_size'    : self.kernel_size
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)
    
class WaveGlow(tf.keras.Model):
    """
        This class is a perfect copy of the NVIDIA's WaveGlow implementation
            See https://github.com/NVIDIA/waveglow repository
    """
    def __init__(self,
                 n_mel_channels = 80,
                 n_flows        = 12,
                 n_group        = 8,
                 
                 n_early_every  = 4,
                 n_early_size   = 2,
                 
                 n_layers       = 8,
                 n_channels     = 512,
                 kernel_size    = 3,
                 ** kwargs
                ):
        super().__init__(** kwargs)
        assert(n_group % 2 == 0)

        self.n_mel_channels = n_mel_channels
        self.n_flows        = n_flows
        self.n_group        = n_group
        
        self.n_early_every  = n_early_every
        self.n_early_size   = n_early_size
        
        self.n_layers       = n_layers
        self.n_channels     = n_channels
        self.kernel_size    = kernel_size

        self.upsample = tf.keras.layers.Conv1DTranspose(
            n_mel_channels, 1024, strides = 256
        )
        self.blocks     = []
        self.convinv    = []

        n_half = int(n_group / 2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.blocks.append(WaveglowBlock(
                n_in_channels   = n_half,
                n_mel_channels  = n_mel_channels * n_group,
                n_layers    = n_layers,
                n_channels  = n_channels,
                kernel_size = kernel_size
            ))
        
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    @property
    def dummy_inputs(self):
        return tf.random.normal((1, 64, self.n_mel_channels))
    
    def call(self, inputs, training = False):
        return self.infer(inputs)
        #raise NotImplementedError()
    
    #@tf.function(reduce_retracing = True)
    def infer(self, spect, sigma = 1.0):
        spect = self.upsample(spect)

        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.strides[0]
        spect = spect[:, :-time_cutoff, :]

        spect = tf.image.extract_patches(
            tf.expand_dims(spect, -1),
            sizes   = [1, self.n_group,1,1],
            strides = [1,self.n_group,1,1],
            rates   = [1,1,1,1],
            padding = 'VALID'
        )
        spect = tf.reshape(spect, [tf.shape(spect)[0], tf.shape(spect)[1], -1])

        audio = sigma * tf.random.normal([
            tf.shape(spect)[0], tf.shape(spect)[1], self.n_remaining_channels
        ])

        for i, k in enumerate(reversed(range(self.n_flows))):
            n_half  = tf.cast(tf.shape(audio)[2] / 2, tf.int32)
            audio_0 = audio[:, :, :n_half]
            audio_1 = audio[:, :, n_half:]

            output = self.blocks[k]((audio_0, spect))
            
            s = output[:, :, n_half:]
            b = output[:, :, :n_half]

            audio_1 = (audio_1 - b) / tf.exp(s)
            audio   = tf.concat([audio_0, audio_1], axis = 2)
            
            audio = self.convinv[k](audio, reverse = True)

            if k % self.n_early_every == 0 and k > 0:
                z = tf.random.normal([
                    tf.shape(spect)[0], tf.shape(spect)[1], self.n_early_size
                ])
                audio = tf.concat([sigma*z, audio], axis = 2)
        
        return tf.reshape(audio, [tf.shape(audio)[0], -1])
    
    def get_config(self):
        config = {
            'n_mel_channels'    : self.n_mel_channels,
            'n_flows'        : self.n_flows,
            'n_group'        : self.n_group,
        
            'n_early_every'  : self.n_early_every,
            'n_early_size'   : self.n_early_size,
        
            'n_layers'       : self.n_layers,
            'n_channels'     : self.n_channels,
            'kernel_size'    : self.kernel_size
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

def pytorch_waveglow(to_gpu = True, eval_mode = True):
    import torch

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    if eval_mode: waveglow.eval()
    if to_gpu: waveglow = waveglow.to('cuda')
    
    return waveglow

custom_functions    = {
    'nvidia_waveglow'   : pytorch_waveglow,

    'WaveGlow'    : WaveGlow
}

custom_objects  = {
    'Invertible1x1Conv' : Invertible1x1Conv,
    'WaveglowBlock' : WaveglowBlock,
    'WaveGlow'      : WaveGlow
}