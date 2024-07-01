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

from utils.keras_utils import ops

def random_mask(tokens, mask_token_idx, n = 1, min_idx = 0, max_idx = None):
    """ Replaces `n` random values in `tokens[min_idx : max_idx]` by `mask_token_idx` """
    if n == 0: return tokens
    
    n = ops.convert_to_tensor(n)
    if ops.is_float(n): n = ops.cast(n * len(tokens), 'int32')
    
    if max_idx is None: max_idx = len(tokens)
    if max_idx < 0:     max_idx = len(tokens) + max_idx
    
    indices = ops.random.shuffle(ops.arange(min_idx, max_idx, dtype = 'int32'))[: n, None]
    return ops.scatter_update(
        tokens, indices, ops.full((n, ), mask_token_idx, dtype = tokens.dtype)
    )
