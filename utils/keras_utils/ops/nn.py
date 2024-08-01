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

from functools import wraps

from .ops_builder import build_op

__keras_all__ = ['average_pool', 'batch_normalization', 'binary_crossentropy', 'categorical_crossentropy', 'conv', 'conv_transpose', 'ctc_decode', 'ctc_loss', 'depthwise_conv', 'elu', 'gelu', 'hard_sigmoid', 'hard_silu', 'hard_swish', 'leaky_relu', 'log_sigmoid', 'log_softmax', 'max_pool', 'moments', 'multi_hot', 'one_hot', 'psnr', 'relu', 'relu6', 'selu', 'separable_conv', 'sigmoid', 'silu', 'swish', 'softmax', 'softplus', 'softsign', 'sparse_categorical_crossentropy']

globals().update({k : build_op(k, disable_np = True) for k in __keras_all__})

def _tf_conv(name):
    def func(* args, ** kwargs):
        import tensorflow as tf
        if 'padding' in kwargs: kwargs['padding'] = kwargs['padding'].upper()
        return getattr(tf.nn, name)(* args, ** kwargs)
    return func

conv1d  = build_op('conv', _tf_conv('conv1d'), disable_np = True)
conv2d  = build_op('conv', _tf_conv('conv2d'), disable_np = True)
