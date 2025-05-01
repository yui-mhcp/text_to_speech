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

import inspect
import functools

def copy_methods(attr_name, * args, type = None, ** kwargs):
    kwargs.update({name : name for name in args})
    
    def wrapper(cls):
        for cls_fn_name, attr_fn_name in kwargs.items():
            if hasattr(cls, cls_fn_name): continue
            setattr(cls, cls_fn_name, _forward_attribute(attr_name, attr_fn_name, type))
        return cls
    
    return wrapper

def partial(fn = None, * _args, _force = False, _update_doc = False, ** _kwargs):
    """
        Wraps `fn` with default args or kwargs. It acts similarly to `functools.wraps` while adding additional information to `fn.__doc__` by updating the signature
        The major difference with `functools.partial` is that it returns a function instead of an object ! This makes it suitable to wrap methods !
        
        Arguments :
            - fn    : the function to call
            - * _args   : values for the first positional arguments
            - _force    : whether to force kwargs `_kwargs` or not
            - ** _kwargs    : new default values for keyword arguments
        Return :
            - wrapped function (if `fn` is provided) or a decorator
        
        Note that partial positional arguments are only usable in the non-decorator mode (which is also the case in `functools.partial`)
        
        **IMPORTANT** : the below examples can be executed with `functools.partial`, and will give the exact same behavior
        
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
        if not _args and not _kwargs:
            return fn
        elif _args and _kwargs:
            def inner(* args, ** kwargs):
                args = (_args + args) if not add_self_first else (args[:1] + _args + args[1:])

                if _force:
                    kwargs.update(_kwargs)
                else:
                    kwargs = {** _kwargs, ** kwargs}
                return fn(* args, ** kwargs)
        elif _args:
            def inner(* args, ** kwargs):
                args = (_args + args) if not add_self_first else (args[:1] + _args + args[1:])
                return fn(* args, ** kwargs)
        else:
            def inner(* args, ** kwargs):
                if _force:
                    kwargs.update(_kwargs)
                else:
                    kwargs = {** _kwargs, ** kwargs}
                return fn(* args, ** kwargs)
        
        if _update_doc:
            inner = functools.wraps(fn)(inner)
            inner.__doc__ = '{}{}{}'.format(
                '' if not _args else 'partial args : {} '.format(_args),
                '' if not _kwargs else 'partial kwargs : {}'.format(_kwargs),
                '\n{}'.format(inner.__doc__) if inner.__doc__ else ''
            ).strip().capitalize()
            _update_signature(inner, * _args, ** _kwargs)
        
        inner.func  = fn
        inner.args  = _args
        inner.kwargs    = _kwargs
        
        fn_arg_names    = list(inspect.signature(fn).parameters.keys())
        add_self_first  = fn_arg_names and fn_arg_names[0] in ('self', 'cls')
        
        return inner
    return wrapper if fn is None else wrapper(fn)

def dispatch_wrapper(methods, name, default = None):
    def wrapper(fn):
        def dispatch(dispatch_fn = None, keys = None):
            def _wrapper(dispatch_fn):
                methods.update({k : dispatch_fn for k in keys})
                
                _add_dispatch_doc(fn, dispatch_fn, name, keys)
                return dispatch_fn
            
            if not keys:
                if inspect.isfunction(dispatch_fn):
                    keys = dispatch_fn.__name__.split('_')[-1]
                else:
                    dispatch_fn, keys = None, dispatch_fn
            
            if isinstance(keys, str): keys = [keys]
            
            assert keys
            return _wrapper if dispatch_fn is None else _wrapper(dispatch_fn)
        
        fn.dispatch = dispatch
        fn.methods  = methods
        fn.dispatch_arg = name
        
        for method_name, method_fn in sorted(methods.items()):
            fn.dispatch(method_fn, method_name)
        
        return fn
    return wrapper

def _add_dispatch_doc(fn, dispatch_fn, arg_name, keys, default = None):
    if len(keys) == 1: keys = keys[0]
    if isinstance(dispatch_fn, functools.partial): dispatch_fn = dispatch_fn.func
    
    doc = (fn.__doc__ + '\n\n') if fn.__doc__ else ''
    doc += '{}{} : {}\n'.format(
        arg_name, ' (default)' if default and default in keys else '', keys
    )
    doc += '  {}{}'.format(dispatch_fn.__name__, inspect.signature(dispatch_fn))
    fn.__doc__ = doc

def _forward_attribute(attr_name, fn_name, attr_type = None):
    def proxy(self, * args, ** kwargs):
        """ Return `{fn_name}` from `self.{attr_name}` """
        attr = getattr(getattr(self, attr_name), fn_name)
        return attr(* args, ** kwargs) if inspect.ismethod(attr) else attr

    if attr_type is not None:
        if hasattr(attr_type, fn_name):
            proxy = functools.wraps(getattr(attr_type, fn_name))(proxy)
        
        if not hasattr(attr_type, fn_name) or isinstance(getattr(attr_type, fn_name), (property, functools.cached_property)):
            proxy.__name__ = fn_name
            proxy = property(proxy)
    else:
        proxy.__name__ = fn_name
        proxy.__doc__ = proxy.__doc__.format(attr_name = attr_name, fn_name = fn_name)
    
    return proxy

def _update_signature(fn, * args, ** kwargs):
    signature = inspect.signature(fn).parameters.copy()
    
    if args:
        names = list(signature.keys())
        if names and names[0] in ('self', 'cls'): names = names[1:]
        for name in names[:len(args)]: signature.pop(name)
    
    params, is_kw_only, var_kw  = [], False, None
    for k, v in signature.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            var_kw = v
        elif k in kwargs:
            params.append(v.replace(default = kwargs[k], kind = inspect.Parameter.KEYWORD_ONLY))
            is_kw_only = True
        elif is_kw_only and v.kind != inspect.Parameter.KEYWORD_ONLY:
            params.append(v.replace(kind = inspect.Parameter.KEYWORD_ONLY))
        else:
            params.append(v)
    
    for k, v in kwargs.items():
        if k not in signature:
            if var_kw is None:
                raise TypeError('Got an unexpected argument : {}'.format(k))
            
            params.append(inspect.Parameter(
                name = k, default = v, kind = inspect.Parameter.KEYWORD_ONLY
            ))

    if var_kw: params.append(var_kw)

    fn.__signature__ = inspect.Signature(
        params, return_annotation = inspect.signature(fn).return_annotation
    )
