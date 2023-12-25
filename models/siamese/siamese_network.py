# Copyright (C) 2022-now yui-mhcp project's author. All rights reserved.
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
import time
import random
import logging
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer, time_logger
from models.siamese.base_encoder import BaseEncoderModel
from utils.distance import KNN, distance
from utils.embeddings import load_embedding, save_embeddings, embeddings_to_np
from utils import normalize_filename, plot_embedding, pad_batch, sample_df

logger      = logging.getLogger(__name__)

def l2_normalize(x):
    import tensorflow as tf
    return tf.math.l2_normalize(x, axis = -1)

class SiameseNetwork(BaseEncoderModel):
    """
        Base class for Siamese Network architecture 
        
        The concept of Siamese Network is to have a unique model that "encodes" 2 inputs in an embedding space, compare them according to a distance function and based on the distance, decides a score of similarity (or disimilarity).
        
        The output of a Siamese is the output of a Dense with 1 neuron and a sigmoid function. 
        - It means that if embed_distance == True, an output of 0.75 can be interpreted as "75% that the 2 inputs are different"
        - Otherwise if embed_distance == False, an output of 0.75 can beinterpreted as "75% that the 2 inputs are similar (of the same class)"
        
        You must define functions : 
            - build_encoder(** kwargs)  : return a `tf.keras.Sequential` model which will be the encoder of the siamese network
            - get_input(data)       : load a single input data (for the dataset pipeline)
            - augment_input(data)   : augment a single input data
            - preprocess_input(inputs)  : apply a preprocessing on a batch of inputs
        
        The dataset pipeline is as follow : 
            1) encode_data receives a 2-tuple (same, not_same)
                Where both are dict with suffixes _x and _y for data belonging to the 1st / 2nd input
                The get_input(data) receives all key-value pairs with _x and _y suffixes (one at a time) and remove the suffixe so that you can treat it the same way
                `sare` inputs where _x and _y belong to the same class
                `not_same` are inputs where _x and _y belong to different classes
            2) augment_data(same, not_same) : receives input x / y (non-batched) once at a time so that you can augment them in independant way
                Each (from same and not_same) will be passed to `augment_input`
            
            3) preprocess_data(same, not_same)  : receives batched datas for same and not same. They are then concatenated to form a single batch of same and not_same
                Note that if the 1st dimension mismatch (variable length data), they are padded to match the longest one
    """
    def __init__(self,
                 * args,
                 threshold  = 0.5,
                 output_similarity  = True,

                 embed_distance     = None,
                 
                 ** kwargs
                ):
        should_update = False
        if embed_distance is not None:
            should_update   = True
            output_similarity   = not embed_distance
            
        self.threshold  = threshold
        self.output_similarity = output_similarity
        
        super().__init__(* args, ** kwargs)
        
        if should_update: self.save_config()
    
    def _build_model(self, normalize = True, ** kwargs):
        """ Build the `siamese` architecture with self.build_encoder() as encoder part """
        encoder = self.build_encoder(normalize = normalize, ** kwargs)
        
        if isinstance(encoder, tf.keras.Sequential):
            if normalize:
                encoder.add(tf.keras.layers.Lambda(
                    l2_normalize, name = 'normalization_layer'
                ))
            input_kwargs = {'input_shape' : encoder.input_shape[1:]}
        else:
            if normalize:
                logger.warning("Encoder is not a `tf.keras.Sequential` so you have to handle `normalize` internally !")
            input_kwargs = {'input_signature' : self.encoder_input_signature}
        
        siamese_config = {
            'architecture_name' : 'siamese',
            'model'             : encoder,
            'distance_metric'   : self.distance_metric,
            ** input_kwargs
        }
        
        super(BaseEncoderModel, self)._build_model(siamese = siamese_config)
    
    @property
    def input_signature(self):
        return (self.encoder_input_signature, self.encoder_input_signature)
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None, 1), dtype = tf.int32)
    
    @property
    def encoder(self):
        signature = self.encoder_input_signature
        n = 1 if not isinstance(signature, (list, tuple)) else len(signature)
        return self.siamese.layers[n * 2]
    
    @property
    def decoder(self):
        signature = self.encoder_input_signature
        n = 1 if not isinstance(signature, (list, tuple)) else len(signature)
        
        inputs = [
            tf.keras.layers.Input(shape = (None, self.embedding_dim)),
            tf.keras.layers.Input(shape = (None, self.embedding_dim))
        ]
        out = inputs
        for l in self.siamese.layers[n * 2 + 1:]: out = l(out)
        return tf.keras.Model(inputs, out, name = 'siamese_decoder')
                
    def compile(self, loss = 'binary_crossentropy', metrics = ['binary_accuracy', 'eer'], ** kwargs):
        super(BaseEncoderModel, self).compile(loss = loss, metrics = metrics, ** kwargs)
    
    def decode_output(self, output):
        """
            Return whether the 2 inputs can be considered as the same class based on the output score 
            
            Arguments :
                - output    : `tf.Tensor` or `np.ndarray`, the output scores
            Return :
                - decision  : boolean array of the same shape as `output` with `True` if the inputs are "same" and `False` otherwise
        """
        return output > self.threshold if self.output_similarity else output < self.threshold

    def encode(self, data):
        """
            Call self.get_input() on normalized version of data (by removing the _x / _y suffix)
            This function process separately `same` and `not_same`
        """
        inp_x = self.get_input({k[:-2] : v for k, v in data.items() if '_x' in k})
        inp_y = self.get_input({k[:-2] : v for k, v in data.items() if '_y' in k})
        
        if 'same' in data: same = int(data['same'])
        elif 'id' in data: same = 1
        else: same = int(data['id_x'] == data['id_y'])
        
        return (inp_x, inp_y), [same]
    
    def filter(self, data):
        (inp_x, inp_y), target = data
        return tf.logical_and(self.filter_input(inp_x), self.filter_input(inp_y))
    
    def augment(self, data):
        """ Augment `same` or `not_same` separately """
        (inp_x, inp_y), target = data
        
        inp_x = self.augment_input(inp_x)
        inp_y = self.augment_input(inp_y)
        
        return (inp_x, inp_y), target
    
    def concat(self, x_same, x_not_same):
        """
            Concat both batched `x_same` and `x_not_same` together (same function called for the y) 
        """
        return tf.concat([x_same, x_not_same], axis = 0)

    def encode_data(self, same, not_same):
        """ Apply `self.encode()` on same and not_same separately """
        return self.encode(same), self.encode(not_same)
    
    def filter_data(self, same, not_same):
        return tf.logical_and(self.filter(same), self.filter(not_same))
    
    def augment_data(self, same, not_same):
        """ Apply `self.augment()` on same and not_same separately """
        return self.augment(same), self.augment(not_same)
    
    def preprocess_data(self, same, not_same):
        """ 
            oncat `x` and `y` from same and not_same and call `self.preprocess_input` on the batched result
            
            In theory it should also pad inputs but it is not working (quite strange...)
            As a solution, I created the `self.concat(x, y)` method so that you can pad as you want (can be useful if you have special padding_values)
        """
        (same_x, same_y), target_same = same
        (not_same_x, not_same_y), target_not_same = not_same
        
        inp_x = self.preprocess_input(self.concat(same_x, not_same_x))
        inp_y = self.preprocess_input(self.concat(same_y, not_same_y))
        
        target = tf.concat([target_same, target_not_same], axis = 0)
        if not self.output_similarity: target = 1 - target
        
        return (inp_x, inp_y), target
        
    def get_dataset_config(self, ** kwargs):
        """ Add default configuration for siamese dataset """
        kwargs.update({'batch_before_map' : True, 'siamese' : True})
        
        return super().get_dataset_config(** kwargs)
        
    def predict_with_target(self, batch, step, prefix, directory = None, **kwargs):
        """
            Embed the x / y in batch and plot their embeddings 
            This function should be improved to add labels information but as ids are not in batch, I do not know how to add information of similarity in the plots...
        """
        if directory is None: directory = self.train_test_dir
        else: os.makedirs(directory, exist_ok = True)
        kwargs.setdefault('show', False)
        
        (inp_x, inp_y), _ = batch
        
        encoder     = self.encoder
        embedded_x  = encoder(inp_x, training = False)
        embedded_y  = encoder(inp_y, training = False)
        embedded    = tf.concat([embedded_x, embedded_y], axis = 0)
        
        title       = 'embedding space (step {})'.format(step)
        filename    = os.path.join(directory, prefix + '.png')
        plot_embedding(
            embedded, filename = filename, title = title, ** kwargs
        )
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            'threshold' : self.threshold,
            'output_similarity' : self.output_similarity
        })
        return config
    
