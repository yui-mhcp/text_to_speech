
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import inspect
import functools

logger = logging.getLogger(__name__)

class ContextManager:
    def __init__(self, enter = None, exit = None):
        self.enter  = enter
        self.exit   = exit
    
    def __enter__(self):
        return self.enter() if self.enter is not None else None
    
    def __exit__(self, * args):
        return self.exit(* args) if self.exit is not None else None
        
def time_to_string(seconds):
    """ Returns a string representation of a time (given in seconds) """
    if seconds < 0.001: return '{} \u03BCs'.format(int(seconds * 1000000))
    if seconds < 1.:    return '{} ms'.format(int(seconds * 1000))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = ((seconds % 3600) % 60)
    
    return '{}{}{}'.format(
        '' if h == 0 else '{}h '.format(h),
        '' if m == 0 else '{}min '.format(m),
        '{:.3f} sec'.format(s) if m + h == 0 else '{}sec'.format(int(s))
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

def partial(fn = None, * partial_args, _force = False, ** partial_config):
    """
        Wraps `fn` with default args or kwargs. It acts similarly to `functools.wraps` but gives cleaner doc information
        The major difference with `functools.partial` is that it supports class methods !
        
        Arguments :
            - fn    : the function to call
            - * partial_args    : values for the first positional arguments
            - _force    : whether to force kwargs `partial_config` or not
            - ** partial_config : new default values for keyword arguments
        Return :
            - wrapped function (if `fn`) or a decorator
        
        Note that partial positional arguments are only usable in the non-decorator mode (which is also the case in `functools.partial`)
        
        /!\ IMPORTANT : the below examples can be executed with `functools.partial` and will give the exact same behavior
        
        Example :
        ```python
        @partial(b = 3)
        def foo(a, b = 2):
            return a ** b
        
        print(foo(2))   # displays 8 as `b`'s default value is now 3
        
        # In this case, the `a` parameter is forced to be 2 and will be treated as a constant value
        # passing positional arguments to `partial` is a bit risky, but it can be useful in some scenario if you want to *remove* some parameters and fix them to a constant
        # Example : `logging.dev = partial(logging.log, DEV)` --> `logging.dev('message')`
        # Example : `logging.Logger.dev = partial(logging.Logger.log, DEV)`
        # --> `logging.getLogger(__name__).dev('message')` This will not work with functools.partial
        foo2 = partial(foo, 2)

        print(foo2())   # displays 8 as `a`'s value has been set to 2
        print(foo2(b = 4)) # displays 16
        print(foo2(4))  # raises an exception because `b` is given as positional arguments (4) and keyword argument (3)
        
        # In this case, foo3 **must** be called with kwargs, otherwise it will raise an exception
        foo3 = partial(a = 2, b = 2)
        ```
    """
    def wrapper(fn):
        @functools.wraps(fn)
        def inner(* args, ** kwargs):
            if not add_self_first:
                args    = partial_args + args
            else:
                args    = args[:1] + partial_args + args[1:]
            kwargs  = {** kwargs, ** partial_config} if _force else {** partial_config, ** kwargs}
            return fn(* args, ** kwargs)
        
        inner.__doc__ = '{}{}{}'.format(
            '' if not partial_args else 'partial args : {} '.format(partial_args),
            '' if not partial_config else 'partial config : {}'.format(partial_config),
            '\n{}'.format(inner.__doc__) if inner.__doc__ else ''
        ).capitalize()
        
        fn_arg_names    = inspect.getfullargspec(fn).args
        add_self_first  = fn_arg_names and fn_arg_names[0] == 'self'
        
        return inner
    
    return wrapper if fn is None else wrapper(fn)
