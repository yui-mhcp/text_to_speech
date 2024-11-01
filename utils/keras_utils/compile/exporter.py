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

import keras
import inspect
import logging

from keras import tree
from functools import partial, wraps

logger = logging.getLogger(__name__)

def export_function(fn, directory, *, signatures = None, endpoints = None, ** kwargs):
    assert signatures is not None or endpoints
    
    if not endpoints: endpoints = {'serve' : signatures}
    
    archive = keras.export.ExportArchive()
    for endpoint, signature in endpoints.items():
        if not isinstance(signature, (list, tuple, dict)): signature = [signature]
        signature = tree.map_structure(_get_tf_spec, signature)
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Computing function for endpoint {} with signature : {}'.format(
                endpoint, signature
            ))
        
        compiled_fn = fn
        
        if isinstance(signature, dict) and any(not hasattr(v, 'shape') for v in signature.values()):
            static      = {k : v for k, v in signature.items() if not hasattr(v, 'shape')}
            signature   = {k : v for k, v in signature.items() if k not in static}

            compiled_fn = partial(compiled_fn, ** static)
        
        if isinstance(signature, dict):
            keys, values = list(zip(* signature.items()))
            compiled_fn = redirection_wrapper(compiled_fn, keys)
            signature = list(values)
        
        archive.add_endpoint(name = endpoint, fn = compiled_fn, input_signature = signature)

    return archive.write_out(directory)

def redirection_wrapper(fn, keys):
    @wraps(fn)
    def wrapped(* args, ** kwargs):
        kwargs.update({name : arg for name, arg in zip(keys, args)})
        return fn(** kwargs)
    
    wrapped.__signature__ = inspect.Signature([
        inspect.Parameter(name = name, kind = inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in keys
    ])

    return wrapped

def _get_tf_spec(value):
    if not hasattr(value, 'shape'): return value
    
    import tensorflow as tf
    
    dtype = value.dtype
    if hasattr(dtype, 'name'): dtype = dtype.name
    elif dtype is None: return tf.TensorShape(value.shape)
    return tf.TensorSpec(shape = value.shape, dtype = dtype)
