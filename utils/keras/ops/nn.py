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

from functools import cache, wraps

from .builder import Ops

__all__ = ['conv1d', 'conv2d']

@cache
def __getattr__(name):
    if name not in globals():
        return Ops(name, submodule = 'nn', disable_np = True)
    return globals()[name]

def _tf_conv(name):
    def func(* args, ** kwargs):
        import tensorflow as tf
        if 'padding' in kwargs: kwargs['padding'] = kwargs['padding'].upper()
        return getattr(tf.nn, name)(* args, ** kwargs)
    return func

conv1d  = Ops('conv', tensorflow_fn = _tf_conv('conv1d'), disable_np = True)
conv2d  = Ops('conv', tensorflow_fn = _tf_conv('conv2d'), disable_np = True)