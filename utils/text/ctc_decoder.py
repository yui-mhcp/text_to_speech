# Copyright (C) 2022-now yui-mhcp project author. All rights reserved.
# Licenced under a modified Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import keras.ops as K

from utils.keras_utils import TensorSpec, ops, graph_compile
from utils.wrapper_utils import dispatch_wrapper
from utils.sequence_utils import pad_batch

_inf = float('inf')

_ctc_decoder_methods    = {}

@dispatch_wrapper(_ctc_decoder_methods, 'method')
def ctc_decode(sequence, lengths = None, blank_index = 0, method = 'greedy', ** kwargs):
    if method not in _ctc_decoder_methods:
        raise ValueError("Unknown CTC method !\n  Accepted : {}\n  Got : {}".format(
            tuple(_ctc_decoder_methods.keys()), method
        ))
    
    if len(ops.shape(sequence)) == 2: sequence = ops.expand_dims(sequence, axis = 0)
    
    if lengths is None:
        lengths = ops.fill((len(sequence), ), ops.shape(sequence)[1])
    else:
        lengths = ops.cast(lengths, 'int32')
        if len(ops.shape(lengths)) == 0: lengths = ops.expand_dims(lengths, axis = 0)
    
    return _ctc_decoder_methods[method](
        sequence, lengths = lengths, blank_index = blank_index
    )

@ctc_decode.dispatch('greedy')
@graph_compile(input_signature = [
    TensorSpec(shape = (None, None, None), dtype = 'float32'),
    TensorSpec(shape = (None, ), dtype = 'int32')
])
def ctc_greedy_decoder(sequence, lengths, blank_index):
    tokens, scores = K.ctc_decode(
        sequence, lengths, strategy = 'greedy', mask_index = blank_index
    )
    return tokens[0], scores[:, 0] / K.cast(lengths, scores.dtype)

@ctc_decode.dispatch(('beam', 'beam_search'))
@graph_compile(input_signature = [
    TensorSpec(shape = (None, None, None), dtype = 'float32'),
    TensorSpec(shape = (None, ), dtype = 'int32')
])
def tf_ctc_beam_search_decoder(sequence, lengths, blank_index = 0):
    tokens, scores = K.ctc_decode(
        sequence, lengths, strategy = 'beam_search', mask_index = blank_index
    )
    tokens = K.transpose(tokens, [1, 0, 2])
    return tokens, scores / K.cast(lengths, scores.dtype)[:, None]

def ctc_beam_search_decoder(encoded, lm = {}, blank_idx = 0, beam_width = 25, ** kwargs):
    def build_beam(p_tot = - _inf, p_b = - _inf, p_nb = - _inf, p_text = 0):
        return {'p_tot' : p_tot, 'p_b' : p_b, 'p_nb' : p_nb, 'p_text' : p_text}
    
    if ops.rank(encoded) == 3:
        return pad_batch([beam_search_decoder(
            enc, lm = lm, blank_idx = blank_idx, beam_width = beam_width
        ) for enc in encoded], pad_value = blank_idx)
    
    beams = {() : build_beam(p_b = 0, p_tot = 0)}
    
    encoded = ops.convert_to_numpy(encoded)
    
    for t in range(len(encoded)):
        new_beams = {}
        
        best_beams = sorted(beams.keys(), key = lambda l: beams[l]['p_tot'], reverse = True)[:beam_width]

        for label in best_beams:
            p_nb = (beams[label]['p_nb'] + encoded[t, label[-1]]) if label else - _inf
            
            p_b = beams[label]['p_tot'] + encoded[t, blank_idx]
            
            p_tot = np.logaddexp(p_nb, p_b)
            
            new_beams[label] = build_beam(
                p_tot  = np.logaddexp(new_beams.get(label, {}).get('p_tot', - _inf), p_tot),
                p_b    = np.logaddexp(new_beams.get(label, {}).get('p_b', - _inf), p_b),
                p_nb   = np.logaddexp(new_beams.get(label, {}).get('p_nb', - _inf), p_nb),
                p_text = beams[label]['p_text']
            )
            
            new_p_nb = beams[label]['p_tot'] + encoded[t]
            if label: new_p_nb[label[-1]] = beams[label]['p_b'] + encoded[t, label[-1]]
            
            for c in range(len(encoded[0])):
                if c == blank_idx: continue
                
                new_label = label + (c, )
                
                new_beams[new_label] = build_beam(
                    p_tot = np.logaddexp(new_beams.get(new_label, {}).get('p_tot', - _inf), new_p_nb[c]),
                    p_nb  = np.logaddexp(new_beams.get(new_label, {}).get('p_nb', - _inf), new_p_nb[c])
                )
                
        beams = new_beams

    best_beam = sorted(beams.items(), key = lambda b: b[1]['p_tot'], reverse = True)[0]
    return np.array(best_beam[0])
