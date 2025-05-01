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

from functools import cache

from .builder import Ops

@cache
def __getattr__(name):
    if name not in globals():
        return Ops(name, submodule = 'image', disable_np = True)
    return globals()[name]

def _tf_image_resize(* args, interpolation = None, ** kwargs):
    import tensorflow as tf
    if interpolation is not None: kwargs['method'] = interpolation
    return tf.image.resize(* args, ** kwargs)

resize  = image_resize  = resize_image  = Ops(
    'resize', submodule = 'image', tensorflow_fn = _tf_image_resize, disable_np = True
)

__all__ = list(k for k, v in globals().items() if isinstance(v, Ops))
