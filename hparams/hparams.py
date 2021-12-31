import logging

from utils import load_json, dump_json, parse_args

class HParams:
    def __init__(self, _prefix = None, ** kwargs):
        self.__prefix   = _prefix
        self.__config   = {}
        self.update(kwargs)

    @property
    def config(self):
        return self.__config
            
    def __str__(self):
        return "HParams :\n- {}".format(
            "\n- ".join(["{}\t: {}".format(k, v) for k, v in self.__config.items()])
        )
    
    def __call__(self, ** kwargs):
        new_params = self.copy()
        new_params.update(kwargs)
        return new_params
        
    def __contains__(self, v):
        v = _remove_prefix(self.__prefix, v)
        return v in self.__config
        
    def __getattr__(self, key):
        if '_HParams__' in key: return object.__getattribute__(self, key)
        key = _remove_prefix(self.__prefix, key)
        if key not in self.__config:
            raise ValueError("{} not in parameters !".format(key))
        return self.__config[key]
    
    def __setattr__(self, key, value):
        if '_HParams__' in key: object.__setattr__(self, key, value)
        else:
            key = _remove_prefix(self.__prefix, key)
            self.__config[key] = value
        
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    def __eq__(self, v):
        if not isinstance(v, (dict, HParams)): return False
        v_config = v if isinstance(v, dict) else v.config
        return self.__prefix == v.__prefix and self.config == v_config
    
    def __add__(self, v):
        if not isinstance(v, (dict, HParams)):
            raise ValueError("V must be dict or HParams instance !")
        
        v_config = v if isinstance(v, dict) else v.get_config(with_prefix = True)
        self_config = self.get_config(with_prefix = True)
        for k in v_config.keys():
            if k in self_config and self[k] != v_config[k]:
                logging.warning("Value {} is present in both HParams with different values ({} vs {}) !".format(k, self[k], v_config[k]))
        
        return HParams(** {** self_config, ** v_config})
    
    def copy(self):
        return HParams(_prefix = self.__prefix, ** self)
    
    def extract(self, values, pop = False):
        """ Update self.config without adding new keys """
        keys = list(values.keys())
        for k in keys:
            if k not in self: continue
            v = values.pop(k) if pop else values.get(k)
            setattr(self, k, v)
        return self
    
    def update(self, v):
        """ update self.config and add new keys if any """
        if not isinstance(v, (dict, HParams)):
            raise ValueError("V must be dict or HParams instance !")
        
        v_config = v if isinstance(v, dict) else v.get_config(with_prefix = True)
        for k, v in v_config.items():
            setattr(self, k, v)
        return self

    def setdefault(self, key, value = None):
        """ Set default value for `key` (id set the value only if not already in) """
        if isinstance(key, (dict, HParams)):
            for k, v in key.items():
                self.setdefault(k, v)
            return
        key = _remove_prefix(self.__prefix, key)
        self.__config.set_default(key, value)
    
    def get(self, key, default = None):
        key = _remove_prefix(self.__prefix, key)
        return self.__config.get(key, default)
    
    def pop(self, key, default = None):
        key = _remove_prefix(self.__prefix, key)
        return self.__config.pop(key, default)
    
    def items(self):
        return self.__config.items()
    
    def keys(self):
        return self.__config.keys()
    
    def values(self):
        return self.__config.values()
    
    def get_config(self, with_prefix = False, prefix = None, add_prefix = None):
        config = self.__config.copy()
        if prefix is not None and prefix != self.__prefix:
            config = {
                _remove_prefix(prefix, k) : v for k, v in config.items() 
                if k.startswith(prefix + '_')
            }
        
        if with_prefix and self.__prefix:
            config = {'{}_{}'.format(self.__prefix, k) : v for k, v in config.items()}
        
        if add_prefix is not None:
            config = {'{}_{}'.format(add_prefix, k) : v for k, v in config.items()}
        
        return config
    
    def parse_args(self, * args, ** kwargs):
        args_config = parse_args(* args, ** {** self.get_config(), ** kwargs})
        self.update(args_config)
        return self
    
    def save(self, filename):
        config = self.get_config(with_prefix = True)
        config['_prefix'] = self.__prefix
        dump_json(filename, config, indent = 4)
    
    @classmethod
    def load(cls, filename):
        return cls(** load_json(filename))

def _remove_prefix(prefix, k):
    if prefix is None: return k
    prefix += '_'
    if not k.startswith(prefix): return k
    return k[len(prefix) :]
