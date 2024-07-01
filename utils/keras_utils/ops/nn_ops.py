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

import keras.ops as K

from functools import wraps

from .ops_builder import build_op

def _tf_conv(name):
    def func(* args, ** kwargs):
        import tensorflow as tf
        if 'padding' in kwargs: kwargs['padding'] = kwargs['padding'].upper()
        return getattr(tf.nn, name)(* args, ** kwargs)
    return func

def _tf_stft(* args, center = None, ** kwargs):
    import tensorflow as tf
    return tf.signal.stft(* args, ** kwargs)

def _tf_image_resize(* args, interpolation = None, ** kwargs):
    import tensorflow as tf
    if interpolation is not None: kwargs['method'] = interpolation
    return tf.image.resize(* args, ** kwargs)

conv    = K.conv
conv1d  = build_op('conv', _tf_conv('conv1d'), disable_np = True)
conv2d  = build_op('conv', _tf_conv('conv2d'), disable_np = True)

stft    = build_op('stft', _tf_stft, disable_np = True)
resize_image    = image_resize  = resize    = build_op(
    'image.resize', _tf_image_resize, disable_np = True
)
rgb_to_grayscale    = build_op(
    'image.rgb_to_grayscale', disable_np = True
)

