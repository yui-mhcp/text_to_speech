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

from keras import layers

from .layers import Invertible1x1Conv

def _add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    in_act = input_a + input_b
    t_act = K.tanh(in_act[:, :, :n_channels])
    s_act = K.sigmoid(in_act[:, :, n_channels:])
    acts = t_act * s_act
    return acts

@keras.saving.register_keras_serializable('waveglow')
class WaveglowBlock(keras.Model):
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
                 fused          = False,
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        
        self.n_in_channels  = n_in_channels
        self.n_mel_channels = n_mel_channels
        self.n_layers       = n_layers
        self.n_channels     = n_channels
        self.kernel_size    = kernel_size
        self.fused          = fused
        
        self.in_layers          = []
        self.res_skip_layers    = []
        self.cond_layers        = []

        self.start = layers.Conv1D(n_channels, 1, name = 'start_conv')

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        self.end = layers.Conv1D(
            2 * n_in_channels, 1, kernel_initializer = 'zeros', name = 'end_conv'
        )
        if self.fused:
            self.cond_layer = layers.Conv1D(
                2 * n_channels * n_layers, 1, name = 'cond_layer'
            )

        for i in range(n_layers):
            dilation = 2 ** i

            self.in_layers.append(layers.Conv1D(
                2 * n_channels,
                kernel_size = kernel_size,
                padding     = 'valid',
                dilation_rate   = dilation,
                name    = f'in_conv-{i}'
            ))
            if not self.fused:
                self.cond_layers.append(layers.Conv1D(2 * n_channels, 1, name = f'cond_layer-{i}'))

            # last one is not necessary
            res_skip_channels = 2 * n_channels if i < n_layers - 1 else n_channels

            self.res_skip_layers.append(layers.Conv1D(
                res_skip_channels, 1, name = f'res_skip_conv-{i}'
            ))

    def build(self, input_shape):
        super().build(input_shape)
        audio_shape, spect_shape = input_shape
        
        self.start.build(audio_shape)
        if self.fused:
            self.cond_layer.build(spect_shape)
        
        for i in range(self.n_layers):
            self.in_layers[i].build((None, None, self.n_channels))
            if not self.fused: self.cond_layers[i].build(spect_shape)
            self.res_skip_layers[i].build((None, None, self.n_channels))
        
        self.end.build((None, None, self.n_channels))

    def call(self, forward_input, training = False, mask = None):
        audio, spect = forward_input
        
        audio = self.start(audio)
        if self.fused: spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            dilation = 2 ** i
            pad_half = (self.kernel_size * dilation - dilation) // 2
            
            padding = [(0, 0), (pad_half, pad_half), (0, 0)]
            if self.fused:
                acts = _add_tanh_sigmoid_multiply(
                    self.in_layers[i](K.pad(audio, padding)),
                    spect[:, :, i * 2 * self.n_channels : (i + 1) * 2 * self.n_channels],
                    self.n_channels
                )
            else:
                acts = _add_tanh_sigmoid_multiply(
                    self.in_layers[i](K.pad(audio, padding)),
                    self.cond_layers[i](spect),
                    self.n_channels
                )

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = res_skip_acts[:, :, : self.n_channels] + audio
                skip_acts = res_skip_acts[:, :, self.n_channels:]
            else:
                skip_acts = res_skip_acts

            if i == 0:
                output = skip_acts
            else:
                output = skip_acts + output
        
        return self.end(output)

    def compute_output_shape(self, input_shape):
        return (None, None, 2 * self.n_in_channels)
    
    def get_config(self):
        config = {
            'n_in_channels'     : self.n_in_channels,
            'n_mel_channels'    : self.n_mel_channels,
        
            'n_layers'       : self.n_layers,
            'n_channels'     : self.n_channels,
            'kernel_size'    : self.kernel_size,
            'fused'          : self.fused,
        }
        return config
    
@keras.saving.register_keras_serializable('waveglow')
class WaveGlow(keras.Model):
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

                 fused = False,
                 
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

        self.seed_generator = keras.random.SeedGenerator()
        
        self.upsample = layers.Conv1DTranspose(
            n_mel_channels, 1024, strides = 256, name = 'upsample'
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
            
            self.convinv.append(Invertible1x1Conv(
                n_remaining_channels, name = f'invertible_conv-{k}'
            ))
            self.blocks.append(WaveglowBlock(
                n_in_channels   = n_half,
                n_mel_channels  = n_mel_channels * n_group,
                n_layers    = n_layers,
                n_channels  = n_channels,
                kernel_size = kernel_size,
                fused       = fused,
                name    = f'block-{k}'
            ))
        
        self.n_remaining_channels = n_remaining_channels  # Useful during inference
        self.build((None, None, self.n_mel_channels))
    
    def build(self, input_shape):
        if self.built: return
        super().build(input_shape)
        self.upsample.build(input_shape)
        
        spect_channels = input_shape[2] * self.n_group
        channels = self.n_remaining_channels
        for k in reversed(range(self.n_flows)):
            self.blocks[k].build([(None, None, channels // 2), (None, None, spect_channels)])
            self.convinv[k].build((None, None, channels))
            if k % self.n_early_every == 0 and k > 0:
                channels += self.n_early_size

    def call(self, inputs, z = None, sigma = 1.):
        return self.infer(inputs, z, sigma)
    
    def infer(self, inputs, z = None, sigma = 1.0, deterministic = False):
        spect = self.upsample(inputs)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.strides[0]
        spect = spect[:, :-time_cutoff, :]

        length_spect_group = K.shape(spect)[1] // self.n_group
        spect = K.reshape(spect, (K.shape(spect)[0], length_spect_group, self.n_group, self.n_mel_channels))
        spect = K.transpose(spect, [0, 1, 3, 2])
        spect = K.reshape(spect, (K.shape(spect)[0], length_spect_group, self.n_group * self.n_mel_channels))

        #spect = keras.ops.image.extract_patches(
        #    K.expand_dims(spect, -1),
        #    size    = (self.n_group, 1),
        #    strides = (self.n_group, 1),
        #    dilation_rate   = 1,
        #    padding = 'valid'
        #)
        #spect = K.reshape(spect, [K.shape(spect)[0], K.shape(spect)[1], -1])
        
        if deterministic:
            noise = keras.ops.zeros([
                K.shape(spect)[0], K.shape(spect)[1], self.n_remaining_channels
            ])
        elif z is not None:
            noise = z[:, :, :self.n_remaining_channels]
            z = z[:, :, self.n_remaining_channels : self.n_group]
        else:
            noise = keras.random.normal([
                K.shape(spect)[0], K.shape(spect)[1], self.n_remaining_channels
            ], seed = self.seed_generator)
        audio = sigma * noise

        for i, k in enumerate(reversed(range(self.n_flows))):
            n_half  = K.shape(audio)[2] // 2
            audio_0 = audio[:, :, :n_half]
            audio_1 = audio[:, :, n_half:]

            output = self.blocks[k]((audio_0, spect))
            
            s = output[:, :, n_half:]
            b = output[:, :, :n_half]

            audio_1 = (audio_1 - b) / K.exp(s)
            audio   = K.concatenate([audio_0, audio_1], axis = 2)
            
            audio = self.convinv[k](audio, reverse = True)

            if k % self.n_early_every == 0 and k > 0:
                if deterministic:
                    z_i = keras.ops.zeros([
                        K.shape(spect)[0], K.shape(spect)[1], self.n_early_size
                    ])
                elif z is not None:
                    z_i = z[:, :, : self.n_early_size]
                    z = z[:, :, self.n_early_size : self.n_group]
                else:
                    z_i = keras.random.normal([
                        K.shape(spect)[0], K.shape(spect)[1], self.n_early_size
                    ], seed = self.seed_generator)
                audio = K.concatenate([sigma * z_i, audio], axis = 2)
        
        return K.reshape(audio, [K.shape(audio)[0], -1])
    
    def set_weights(self, weights, ** kwargs):
        super().set_weights(weights, ** kwargs)
        for l in self.convinv: l.build_inverse()

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

def pytorch_waveglow(to_gpu = True, eval_mode = True):
    import torch

    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    if eval_mode: waveglow.eval()
    if to_gpu: waveglow = waveglow.to('cuda')
    
    return waveglow
