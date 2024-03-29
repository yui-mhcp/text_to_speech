
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

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from custom_architectures import get_architecture
from models.siamese.siamese_network import SiameseNetwork
from models.interfaces.base_audio_model import BaseAudioModel
from utils import load_json, dump_json, normalize_filename
from utils.audio import load_audio, write_audio, AudioAnnotation
from utils.distance import find_clusters

logger      = logging.getLogger(__name__)
time_logger = logging.getLogger('timer')

_supported_encoder_types    = ('rnn', 'conv1d', 'conv2d', 'transformer', 'prenet_transformer')

MIN_AUDIO_TIME      = 0.1 # below 0.1sec the encoding is not really relevant

DEFAULT_MEL_FRAMES      = 80
DEFAULT_AUDIO_RATE      = 16000
DEFAULT_MAX_AUDIO_TIME  = 3

class AudioSiamese(BaseAudioModel, SiameseNetwork):
    def __init__(self,
                 audio_rate     = DEFAULT_AUDIO_RATE,
                 
                 encoder_type       = 'conv1d',
                 max_audio_time     = DEFAULT_MAX_AUDIO_TIME,
                 use_fixed_length_input = False,
                 
                 ** kwargs
                ):
        encoder_type = encoder_type.lower()
        assert encoder_type in _supported_encoder_types
        
        self._init_audio(audio_rate = audio_rate, ** kwargs)
        
        self.encoder_type   = encoder_type
        
        self.max_audio_time     = max_audio_time
        self.use_fixed_length_input = use_fixed_length_input
        
        super().__init__(** kwargs)
    
    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.identification_dir, exist_ok = True)

    def build_encoder(self, 
                      depth             = 128,
                      flatten_type      = 'max',
                      embedding_dim     = 128, 
                      flatten_kwargs    = {},
                      normalize         = None,
                      **kwargs
                     ):
        flatten_kwargs.setdefault('units', embedding_dim)
        audio_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 5,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                [depth * 3, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [4, 4, 2, 2, 1],
            'kernel_size'   : [
                32, 32, 3, 3, [3, 1]
            ],
            'residual'      : True,
            'drop_rate'     : 0.05,
            'conv_type'     : self.encoder_type,
            'flatten'       : True,
            'dense_as_final'    : True,
            'name'  : 'Encoder'
        }
        voicemap_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2,
                depth * 3, 
                depth * 4
            ],
            'strides'   : [4, 1, 1, 1],
            'kernel_size'   : [
                32, 3, 3, 3
            ],
            'pooling'       : 'max',
            'pool_size'     : [4, 2, 2, 2],
            'pool_strides'  : [4, 2, 2, 2],
            'residual'      : False,
            'activation'    : 'relu',
            'drop_rate'     : 0.05,
            'conv_type'     : 'conv1d',
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : True,
            'name'  : 'Encoder',
            ** kwargs
        }
        mel_encoder_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.encoder_input_shape[1:],
            'output_shape'  : embedding_dim,
            'n_conv'    : 4,
            'filters'   : [
                depth, 
                depth * 2, 
                [depth * 2, depth * 4],
                [depth * 4, depth * 4],
            ],
            'strides'   : [2, 2, 1, 1],
            'kernel_size'   : [
                7, 5, 3, [3, 1]
            ],
            'residual'      : False,
            'conv_type'     : self.encoder_type,
            'flatten'       : True,
            'flatten_type'  : flatten_type,
            'flatten_kwargs'    : flatten_kwargs,
            'dense_as_final'    : flatten_type not in ('lstm', 'gru'),
            'name'  : 'Encoder',
            ** kwargs
        }
        
        encoder_config = mel_encoder_config if self.use_mel_fn else voicemap_encoder_config
        
        return get_architecture(** encoder_config)

    @property
    def identification_dir(self):
        return os.path.join(self.folder, 'identification')
    
    @property
    def encoder_input_shape(self):
        length = None if not self.use_fixed_length_input else self.max_input_length
        
        if not self.use_mel_fn:
            shape = (None, length, 1)
        elif not self.mel_as_image:
            shape = (None, length, self.n_mel_channels)
        else:
            shape = (None, length, self.n_mel_channels, 1)
        return shape
    
    @property
    def min_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(MIN_AUDIO_TIME * self.audio_rate))
        else:
            return int(MIN_AUDIO_TIME * self.audio_rate)
        
    @property
    def max_input_length(self):
        if self.use_mel_fn:
            return self.mel_fn.get_length(int(self.max_audio_time * self.audio_rate))
        else:
            return int(self.max_audio_time * self.audio_rate)
                
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_audio)
        
    def __str__(self):
        return super().__str__() + self._str_audio()
    
    def get_input(self, data, ** kwargs):
        if isinstance(data, list):
            return [self.get_input(data_i, ** kwargs) for data_i in data]
        elif isinstance(data, pd.DataFrame):
            return [self.get_input(row, ** kwargs) for idx, row in data.iterrows()]
        
        input_data = self.get_audio(data)
        
        if tf.shape(input_data)[0] > self.max_input_length:
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(input_data)[0] - self.max_input_length,
                dtype = tf.int32
            )
            input_data = input_data[start : start + self.max_input_length]
        
        return input_data
    
    
    def augment_input(self, inp):
        return self.augment_audio(inp, max_length = self.max_input_length)
    
    def concat(self, x_same, x_not_same):
        seq_1, seq_2 = tf.shape(x_same)[1], tf.shape(x_not_same)[1]
        
        if seq_1 != seq_2:
            padding = [(0,0), (0, tf.abs(seq_1 - seq_2)), (0,0)]
            if seq_1 > seq_2:
                x_not_same = tf.pad(x_not_same, padding)
            else:
                x_same = tf.pad(x_same, padding)
                
        return tf.concat([x_same, x_not_same], axis = 0)
        
    def get_dataset_config(self, ** kwargs):
        kwargs.update({'pad_kwargs' : {}, 'padded_batch' : True})
        if self.use_fixed_length_input:
            input_shape = self.encoder_input_shape
            kwargs['pad_kwargs'] = {
                'padded_shapes' : (
                    ((input_shape[1:], input_shape[1:]), ()),
                    ((input_shape[1:], input_shape[1:]), ()),
                )
            }
        
        return super().get_dataset_config(** kwargs)
                        
    def embed_dataset(self, * args, ** kwargs):
        kwargs['rate'] = self.audio_rate
        return super().embed_dataset(* args, ** kwargs)
    
    @timer
    def identify(self,
                 filenames,
                 alignments = None,
                 win_len    = 2,
                 step       = 1,
                 
                 batch_size = 128,
                 
                 directory  = None,
                 overwrite  = False,
                 
                 verbose    = True,
                 show   = False,
                 tqdm   = tqdm,
                 
                 ** kwargs
                ):
        filenames = normalize_filename(filenames, invalid_mode = 'keep')
        
        if not isinstance(filenames, (list, tuple)): filenames = [filenames]
        if alignments is not None and not isinstance(alignment, (list, tuple)):
            alignments = [alignments]
        if self.use_fixed_length_input: step = self.max_audio_time
        window = int(win_len * self.audio_rate)
        step   = int(step * self.audio_rate)
        
        ####################
        #  Init directory  #
        ####################
        
        if not directory: directory = self.identification_dir
        audio_dir   = os.path.join(directory, 'audios')
        result_dir  = os.path.join(directory, 'results')
        map_file    = os.path.join(directory, 'map.json')

        os.makedirs(audio_dir, exist_ok = True)
        os.makedirs(result_dir, exist_ok = True)
        
        # Load already predicted data (if any)
        all_outputs = load_json(map_file, default = {})
        
        ########################################
        #  Perform identification on each file #
        ########################################
        
        outputs = []
        for i, filename in enumerate(filenames):
            logger.info("Processing file {}...".format(filename if isinstance(filename, str) else 'raw audio'))
            # Normalize filename and save raw audio (if necessary)
            audio_filename = filename
            if not isinstance(filename, str):
                audio_filename = os.path.join(
                    audio_dir, 'audio_{}.wav'.format(len(os.listdir(audio_dir)))
                )

                write_audio(audio = filename, filename = audio_filename, rate = self.audio_rate)
            elif filename in all_outputs and not overwrite:
                outputs.append(AudioAnnotation.load_from_file(
                    all_outputs[filename]
                ))
                continue
            
            time_logger.start_timer('loading')
            
            audio = filename if not isinstance(filename, str) else load_audio(filename, self.audio_rate)
            
            time_logger.stop_timer('loading')
            
            if alignments is not None:
                alignment = alignments[i]
            else:
                alignment = [{
                    'start' : start / self.audio_rate,
                    'end'   : min(len(audio), start + window) / self.audio_rate, 
                    'time'  : min(window, len(audio) - start) / self.audio_rate
                } for start in range(0, len(audio), step)]
            
            # Embed data
            # Note : embedding `filename` instead of `audio_filename` allows to not
            # reload audio if raw audio was given
            parts = [
                audio[int(align['start'] * self.audio_rate) : int(align['end'] * self.audio_rate)]
                for align in alignment
            ]
            embedded_parts = self.embed(parts, batch_size = batch_size, tqdm = tqdm)
            
            # Build similarity matrix
            #similarity_matrix = self.pred_similarity_matrix(embedded_parts)

            # Predict ids based on the `KPropagation` clustering
            assignment = find_clusters(embedded_parts, k = 3, method = 'kmeans')

            if isinstance(assignment, tuple): assignment = assignment[1]
            if hasattr(assignment, 'numpy'):  assignment = assignment.numpy()

            if plot: plot_embedding(embedded_parts, assignment)

            
            for i, label in enumerate(embedded_parts):
                alignment[i]['id'] = label
            
            # Make time alignment with predicted ids
            result_dir = all_outputs.get(
                audio_filename, 
                os.path.join(result_dir, '{}_result'.format(os.path.basename(filename)[:-4]))
            )
            result = AudioAnnotation(
                directory   = result_dir,
                filename    = audio_filename, 
                rate        = self.audio_rate,
                infos       = alignment,
                ** kwargs
            )
            result.save()

            outputs.append(result)
            all_outputs[audio_filename] = result_dir

            dump_json(map_file, all_outputs, indent = 4)

        return outputs
    
    def get_config(self, *args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_audio(),
            'max_audio_time'    : self.max_audio_time,
            'encoder_type'      : self.encoder_type,
            'use_fixed_length_input'    : self.use_fixed_length_input
        })
            
        return config
