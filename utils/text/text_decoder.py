
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

import numpy as np
import tensorflow as tf

from utils.sequence_utils import pad_batch

_inf = float('inf')

def decode(encoded, method = 'greedy', ** kwargs):
    if method not in _decoder_method:
        raise ValueError("Unknown decoder !\n  Accepted : {}\n  Got : {}".format(
            tuple(_decoder_method.keys()), method
        ))
    
    if len(tf.shape(encoded)) == 2: encoded = tf.expand_dims(encoded, axis = 0)
    
    return _decoder_method[method](encoded, ** kwargs)

def greedy_decoder(encoded, ** kwargs):
    return tf.argmax(encoded, axis = -1)

def tf_beam_search_decoder(encoded, blank_index = 0, return_score = False, ** kwargs):
    tokens, scores = tf.nn.ctc_beam_search_decoder(
        tf.transpose(encoded, [1, 0, 2]),
        tf.fill([tf.shape(encoded)[0]], tf.shape(encoded)[1]),
        ** kwargs
    )

    tokens = tf.sparse.to_dense(tokens[0])
    return tokens if not return_score else (tokens, scores)

def beam_search_decoder(encoded, lm = {}, blank_idx = 0, beam_width = 25, ** kwargs):
    def build_beam(p_tot = - _inf, p_b = - _inf, p_nb = - _inf, p_text = 0):
        return {'p_tot' : p_tot, 'p_b' : p_b, 'p_nb' : p_nb, 'p_text' : p_text}
    
    if tf.rank(encoded) == 3:
        return pad_batch([beam_search_decoder(
            enc, lm = lm, blank_idx = blank_idx, beam_width = beam_width
        ) for enc in encoded], pad_value = blank_idx)
    
    beams = {() : build_beam(p_b = 0, p_tot = 0)}
    
    encoded = encoded.numpy()
    
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

_decoder_method = {
    'greedy'    : greedy_decoder,
    'beam'      : beam_search_decoder,
    'beam_with_lm'  : beam_search_decoder
}