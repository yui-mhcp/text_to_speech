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

from .ops_builder import _is_numpy, build_op, build_custom_op

def _python_while(cond, body, loop_vars, maximum_iterations = float('inf')):
    is_tuple = isinstance(loop_vars, (list, tuple))
    i = 0
    while cond(* loop_vars) and i < maximum_iterations:
        loop_vars = body(* loop_vars)
        if not isinstance(loop_vars, (list, tuple)): loop_vars = [loop_vars]
        i += 1
    return loop_vars if is_tuple else loop_vars[0]

def _check_numpy(args, kwargs, _):
    loop_vars = args[2] if len(args) >= 3 else kwargs['loop_vars']
    return _is_numpy([loop_vars], {}, nested = isinstance(loop_vars, (list, tuple)))

while_loop  = build_op(
    'while_loop', np_op = _python_while, is_numpy_check = _check_numpy
)

def scan_tf(f, init, xs, output = None):
    import tensorflow as tf
    
    def body(state_with_out, inp):
        return f(state_with_out[0], inp)
    
    if output is None:
        _, output = f(init, tf.nest.map_structure(lambda x: x[0], xs))

    states, outputs = tf.scan(body, xs, initializer = (init, output))
    return tf.nest.map_structure(lambda s: s[-1], states), outputs

scan    = build_custom_op(
    tf_fn   = scan_tf,
    jax_fn  = 'lax.scan',
    name    = 'scan'
)
