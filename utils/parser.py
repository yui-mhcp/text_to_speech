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

import os
import inspect
import logging
import argparse
import importlib

logger = logging.getLogger(__name__)

_max_repr_items = 10
_max_repr_str   = 100

def get_fn_name(fn):
    if hasattr(fn, 'name'):         return fn.name
    elif hasattr(fn, '__name__'):   return fn.__name__
    return fn.__class__.__name__

def is_object(o):
    return not isinstance(o, type) and not is_function(o)

def is_function(f):
    return f.__class__.__name__ in ('function', 'method')

def is_bound_method(f):
    return f.__class__.__name__ == 'method'

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

def get_annotations(fn):
    if hasattr(inspect, 'get_annotations'):
        return inspect.get_annotations(fn)
    elif hasattr(fn, '__annotations__'):
        return fn.__annotations__
    else:
        return {}

def has_args(fn, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in inspect.signature(fn, ** kwargs).parameters.values()
    )

def has_kwargs(fn, name = None, ** kwargs):
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD and (name is None or _name == name)
        for _name, param in inspect.signature(fn, ** kwargs).parameters.items()
    )

def signature_to_str(fn, add_doc = False, ** kwargs):
    return '{}{}{}'.format(
        fn.__name__,
        str(inspect.signature(fn, ** kwargs)),
        '\n{}'.format(fn.__doc__) if add_doc else ''
    )

def args_to_str(args = None, kwargs = None):
    if isinstance(args, dict) and kwargs is None: args, kwargs = None, args
    
    if args and kwargs:
        return '{} - {}'.format(args_to_str(args = args), args_to_str(kwargs = kwargs))
    elif args:
        return 'args : {}'.format(repr_data(args))
    elif kwargs:
        return 'kwargs : {}'.format(repr_data(kwargs))

def repr_data(data):
    if hasattr(data, 'shape') and len(data.shape) > 0:
        return '<{} shape={} dtype={}>'.format(data.__class__.__name__, data.shape, data.dtype)
    elif isinstance(data, dict):
        if len(data) <= _max_repr_items:
            return {k : repr_data(v) for k, v in data.items()}
        des = '{'
        for i, (k, v) in enumerate(data.items()):
            if i > _max_repr_items: break
            des += '{} : {}, '.format(k, repr_data(v))
        return des + '... [{} more]}'.format(len(data) - _max_repr_items)

    elif isinstance(arg, (list, tuple)):
        if len(data) <= _max_repr_items:
            return str([repr_data(v) for v in data])
        return '[{}, ... [{} more]]'.format(
            ', '.join([repr_data(v) for v in data[:_max_repr_items]]),
            len(data) - _max_repr_items
        )
    elif isinstance(data, str) and len(data) > _max_repr_str:
        return '{} ... (length {})'.format(data[:_max_repr_str], len(data))
    return repr(data)

def import_objects(modules,
                   exclude  = (),
                   filters  = None,
                   classes  = None,
                   types    = None,
                   err_mode = 'raise',
                   allow_modules    = False,
                   allow_functions  = True,
                   signature    = None,
                   fn   = None
                  ):
    if fn is None: fn = lambda v: v
    def is_valid(name, val, module):
        if not hasattr(val, '__module__'):
            if not allow_modules or not val.__package__.startswith(module): return False
            return True

        if filters is not None and not filters(name, val): return False
        if not isinstance(val, type):
            if types is not None and isinstance(val, types): return True
            
            if not val.__module__.startswith(module): return False
            if allow_functions and callable(val):
                if signature:
                    return get_args(val)[:len(signature)] == signature
                return True
            return False
        if not val.__module__.startswith(module): return False
        if classes is not None and not issubclass(val, classes):
            return False
        return True
            
    if types is not None:
        if not isinstance(types, (list, tuple)): types = (types, )
        if type in types: allow_functions = False
    
    if signature: signature = list(signature)
    if isinstance(exclude, str):        exclude = [exclude]
    if not isinstance(modules, list): modules = [modules]
    
    all_modules = []
    for module in modules:
        if isinstance(module, str):
            all_modules.extend(_expand_path(module))
        else:
            all_modules.append(module)

    objects = {}
    for module in all_modules:
        if isinstance(module, str):
            if module.endswith(('__init__.py', '_old.py', '_new.py')): continue
            elif os.path.basename(module).startswith(('.', '_')): continue
            
            try:
                module = module.replace(os.path.sep, '.')[:-3]
                module = importlib.import_module(module)
            except Exception as e:
                logger.debug('Import of module {} failed due to {}'.format(module, str(e)))
                if err_mode == 'raise': raise e
                continue
        
        root_module = module.__name__.split('.')[0]
        objects.update({
            k : fn(v) for k, v in vars(module).items() if (
                (hasattr(v, '__module__') or hasattr(v, '__package__'))
                and not k.startswith('_')
                and not k in exclude
                and is_valid(k, v, root_module)
            )
        })
    
    return objects

def _expand_path(path):
    expanded = []
    for f in os.listdir(path):
        if f.startswith(('.', '_')): continue
        f = os.path.join(path, f)
        if os.path.isdir(f):
            expanded.extend(_expand_path(f))
        else:
            expanded.append(f)
    return expanded

def parse_args(* args, allow_abrev = True, add_unknown = False, ** kwargs):
    """
        Not tested yet but in theory it parses arguments :D
        Arguments : 
            - args  : the mandatory arguments
            - kwargs    : optional arguments with their default values
            - allow_abrev   : whether to allow abreviations or not (will automatically create abreviations as the 1st letter of the argument if it is the only argument to start with this letter)
    """
    def get_abrev(keys):
        abrev_count = {}
        for k in keys:
            abrev = k[0]
            abrev_count.setdefault(abrev, 0)
            abrev_count[abrev] += 1
        return [k for k, v in abrev_count.items() if v == 1 and k != 'h']
    
    parser = argparse.ArgumentParser()
    for arg in args:
        name, config = arg, {}
        if isinstance(arg, dict):
            name, config = arg.pop('name'), arg
        parser.add_argument(name, ** config)
    
    allowed_abrev = get_abrev(kwargs.keys()) if allow_abrev else {}
    for k, v in kwargs.items():
        abrev = k[0]
        names = ['--{}'.format(k)]
        if abrev in allowed_abrev: names += ['-{}'.format(abrev)]
        
        config = v if isinstance(v, dict) else {'default' : v}
        if not isinstance(v, dict) and v is not None: config['type'] = type(v)
        
        parser.add_argument(* names, ** config)
    
    parsed, unknown = parser.parse_known_args()
    
    parsed_args = {}
    for a in args + tuple(kwargs.keys()): parsed_args[a] = getattr(parsed, a)
    if add_unknown:
        k, v = None, None
        for a in unknown:
            if not a.startswith('--'):
                if k is None:
                    raise ValueError("Unknown argument without key !\n  Got : {}".format(unknown))
                a = var_from_str(a)
                if v is None: v = a
                elif not isinstance(v, list): v = [v, a]
                else: v.append(a)
            else: # startswith '--'
                if k is not None:
                    parsed_args.setdefault(k, v if v is not None else True)
                k, v = a[2:], None
        if k is not None:
            parsed_args.setdefault(k, v if v is not None else True)
    
    return parsed_args
