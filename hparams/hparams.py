
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

from utils import load_json, dump_json, parse_args

logger = logging.getLogger(__name__)

class HParams:
    """ Base class that you can use like a regular `dict` to define models' hyperparameters """
    def __init__(self,
                 *,
                 _name      = 'HParams',
                 _prefix    = None,
                 _propagate = None,
                 _synchronize   = None,
                 ** kwargs
                ):
        self.__name     = _name
        self.__prefix   = _prefix

        self.__config   = {}
        self.update(kwargs)
    
    @property
    def config(self):
        return self.__config
    
    @property
    def prefix(self):
        return self.__prefix
    
    def __str__(self):
        return "{} {}:\n- {}".format(
            self.__name,
            '' if not self.prefix else '(prefix {})'.format(self.prefix),
            "\n- ".join(["{}\t: {}".format(k, v) for k, v in self.config.items()])
        )
    
    def __call__(self, ** kwargs):
        """ Creates a copy of `self` and updates it with `kwargs` """
        new_params = self.copy()
        new_params.update(kwargs)
        return new_params
    
    def __contains__(self, k):
        """ Checks whether the key `k` is in `self.config` """
        return _remove_prefix(self.prefix, k) in self.config
        
    def __getattr__(self, key):
        if '_HParams__' in key: return object.__getattribute__(self, key)
        key = _remove_prefix(self.prefix, key)
        if key not in self.config:
            raise ValueError("{} not in parameters !".format(key))
        return self.config[key]
    
    def __setattr__(self, key, value):
        if '_HParams__' in key: object.__setattr__(self, key, value)
        else: self.config[_remove_prefix(self.prefix, key)] = value
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __eq__(self, v):
        if isinstance(v, dict):
            return self.config == v
        elif isinstance(v, HParams):
            return self.prefix == v.prefix and self.config == v.config
        return False
    
    def __add__(self, v):
        if not isinstance(v, (dict, HParams)):
            raise ValueError("V must be dict or HParams instance !")
        
        v_config = v if isinstance(v, dict) else v.get_config(with_prefix = True)
        self_config = self.get_config(with_prefix = True)
        for k in v_config.keys():
            if k in self_config and self_config[k] != v_config[k]:
                logger.warning("Value {} is present in both HParams with different values ({} vs {}) !".format(k, self_config[k], v_config[k]))
        
        return HParams(** {** self_config, ** v_config})
    
    def copy(self):
        """ Creates a copy of `self` """
        return HParams(_prefix = self.prefix, ** self)
    
    def extract(self, values, pop = False, copy = True):
        """
            Updates `self` (or a copy) without adding new keys
            
            Arguments :
                - value : the key-value mapping to get values from
                - pop   : whether to pop the items from `value`
                - copy  : whether to return a copy of `self` or not
            Returns :
                - `self` if `copy == False` else a new HParams instance
        """
        new_values = {}
        for k in list(values.keys()):
            if k not in self: continue
            new_values[k] = values.pop(k) if pop else values.get(k)
        return self(** new_values) if copy else self.update(new_values)
    
    def update(self, v):
        """ update self.config and add new keys (if any) """
        if not isinstance(v, (dict, HParams)):
            raise ValueError("`v` must be dict or HParams instance !")
        
        v_config = v if isinstance(v, dict) else v.get_config(with_prefix = True)
        for k, v in v_config.items(): self[k] = v
        return self

    def setdefault(self, key, value = None):
        """ Set default value for `key` (id set the value only if not already in) """
        if isinstance(key, (dict, HParams)):
            for k, v in key.items(): self.setdefault(k, v)
            return
        
        self.config.set_default(_remove_prefix(self.prefix, key), value)
    
    def get(self, key, * args):
        return self.config.get(_remove_prefix(self.prefix, key), * args)
    
    def pop(self, key, * args):
        return self.config.pop(_remove_prefix(self.prefix, key), * args)
    
    def items(self):
        return self.config.items()
    
    def keys(self):
        return self.config.keys()
    
    def values(self):
        return self.config.values()
    
    def get_config(self, with_prefix = False, prefix = None, add_prefix = None):
        config = self.config.copy()
        if prefix is not None and prefix != self.prefix:
            config = {
                _remove_prefix(prefix, k) : v for k, v in config.items()
                if k.startswith(prefix + '_')
            }
        
        if with_prefix and self.prefix:
            config = {'{}_{}'.format(self.prefix, k) : v for k, v in config.items()}
        
        if add_prefix is not None:
            config = {'{}_{}'.format(add_prefix, k) : v for k, v in config.items()}
        
        return config
    
    def parse_args(self, * args, ** kwargs):
        args_config = parse_args(* args, ** {** self.get_config(), ** kwargs})
        self.update(args_config)
        return self
    
    def save(self, filename):
        config = self.get_config(with_prefix = True)
        config.update({'_name' : self.__name, '_prefix' : self._prefix})
        dump_json(filename, config, indent = 4)
    
    @classmethod
    def load(cls, filename):
        return cls(** load_json(filename))

def _remove_prefix(prefix, k):
    if prefix is None: return k
    prefix += '_'
    return k if not k.startswith(prefix) else k[len(prefix) :]
