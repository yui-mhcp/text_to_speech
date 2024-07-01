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

import sys
import logging
import inspect
import functools

logger = logging.getLogger(__name__)
        
def time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u03BCs'.format(int(seconds * 1000000))
    if seconds < 0.01:  return '{:.3f} ms'.format(seconds * 1000)
    if seconds < 1.:    return '{} ms'.format(int(seconds * 1000))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = ((seconds % 3600) % 60)
    
    return '{}{}{}'.format(
        '' if h == 0 else '{}h '.format(h),
        '' if m == 0 else '{}min '.format(m),
        '{:.3f} sec'.format(s) if m + h == 0 else '{}sec'.format(int(s))
    )

def executing_eagerly():
    """ This function is equivalent to `tf.executing_eagerly`, while not importing tensorflow by default, removing this heavy dependency if it is not required """
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        return tf.executing_eagerly()
    return True

def get_args(fn, include_args = True, ** kwargs):
    """ Returns a `list` of the positional argument names (even if they have default values) """
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if include_args: kinds += (inspect.Parameter.VAR_POSITIONAL, )
    return [
        name for name, param in inspect.signature(fn, ** kwargs).parameters.items()
        if param.kind in kinds
    ]

def get_kwargs(fn, ** kwargs):
    """ Returns a `dict` containing the kwargs of `fn` """
    return {
        name : param.default for name, param in inspect.signature(fn, ** kwargs).parameters.items()
        if param.default is not inspect._empty
    }

def has_args(fn, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in inspect.signature(fn, ** kwargs).parameters.values()
    )

def has_kwargs(fn, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in inspect.signature(fn, ** kwargs).parameters.values()
    )

def signature_to_str(fn, add_doc = False, ** kwargs):
    return '{}{}{}'.format(
        fn.__name__,
        str(inspect.signature(fn, ** kwargs)),
        '\n{}'.format(fn.__doc__) if add_doc else ''
    )

def get_object(objects, obj, * args, print_name = 'object', err = False, types = None, ** kwargs):
    """
        Get corresponding object based on a name (`obj`) and dict of object names with associated class / function to call (`objects`)
        
        Arguments : 
            - objects   : mapping (`dict`) of names with their associated class / function
            - obj       : the object to build (either a list, str or instance of `types`)
            - args / kwargs : the args and kwargs to pass to the object / function
            - print_name    : name for printing if object is not found
            - err   : whether to raise error if object is not available
            - types : expected return type
        Return : 
            - (list of) instance(s) or function results
    """
    if types is not None and isinstance(obj, types): return obj
    elif obj is None:
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kw
        ) for n, kw in kwargs.items()]
    
    elif isinstance(obj, (list, tuple)):
        return [get_object(
            objects, n, * args, print_name = print_name, err = err, types = types, ** kwargs
        ) for n in obj]
    
    elif isinstance(obj, dict):
        if 'class_name' in obj:
            return get_object(
                objects, obj['class_name'], * args, print_name = print_name, err = err, types = types, ** obj.get('config', {})
            )
        return [get_object(
            objects, n, * args, print_name = print_name,  err = err, types = types, ** kwargs
        ) for n, args in obj.items()]
    
    elif isinstance(obj, str) and obj.lower() in to_lower_keys(objects):
        return to_lower_keys(objects)[obj.lower()](* args, ** kwargs)
    elif err:
        raise ValueError("{} is not available !\n  Accepted : {}\n  Got :{}".format(
            print_name, tuple(objects.keys()), obj
        ))
    else:
        logger.warning("{} : `{}` is not available !".format(print_name, obj))
        return obj

def to_lower_keys(dico):
    return {k.lower() : v for k, v in dico.items()}

