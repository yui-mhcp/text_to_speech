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

from ..keras import TensorSpec, ops, graph_compile

@graph_compile(support_xla = False)
def ctc_decode(sequence : TensorSpec(shape = (None, None, None), dtype = 'float32'),
               lengths  : TensorSpec(shape = (None, ), dtype = 'int32') = None,
               method   = 'greedy',
               blank_index  = 0,
               
               num_beams    : int   = 100,
               num_sentences    : int   = 1
              ):
    if method == 'beam': method = 'beam_search'
    
    if lengths is None:
        lengths = ops.fill((ops.shape(sequence)[0], ), ops.shape(sequence)[1])
    
    tokens, scores = ops.ctc_decode(
        sequence, lengths, strategy = method, mask_index = blank_index, top_paths = num_sentences, beam_width = num_beams
    )
    if method == 'greedy':
        return tokens[0], scores[:, 0] / ops.cast(lengths, scores.dtype)
    elif method == 'beam_search':
        tokens = ops.transpose(tokens, [1, 0, 2])
        return tokens, scores / ops.cast(lengths, scores.dtype)[:, None]

