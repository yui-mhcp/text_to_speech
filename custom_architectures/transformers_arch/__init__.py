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
import glob

from utils.wrapper_utils import partial

_attributes = ('custom_objects', 'custom_functions', '_encoders', '_decoders', '_transformers')

custom_objects = {}
custom_functions = {}

_encoders, _decoders, _transformers = {}, {}, {}

def __load():
    for module_name in glob.glob(os.path.join(* __package__.split('.'), '*.py')):
        if module_name.endswith(('__init__.py', '_old.py')): continue
        module_name = module_name.replace(os.path.sep, '.')[:-3]

        module = __import__(module_name, fromlist = _attributes)
        
        for attr_name in _attributes:
            globals()[attr_name].update(getattr(module, attr_name, {}))

        if not hasattr(module, '_transformers'):
            if hasattr(module, '_encoders'):
                _transformers.update(module._encoders)
            if hasattr(module, '_decoders'):
                _transformers.update(module._decoders)

def __get_pretrained(_cls, _name, pretrained_name, class_name = None, wrapper = None, ** kwargs):
    if wrapper is not None:
        return wrapper.from_pretrained(
            pretrained_name = pretrained_name, class_name = class_name, ** kwargs
        )
    
    if not class_name: class_name = pretrained_name
    class_name  = class_name.lower()
    candidates  = {k.lower() : v for k, v in _cls.items()}
    
    cls = None
    if class_name in candidates:
        cls = candidates[class_name]
    else:
        for name, cand in sorted(candidates.items(), key = lambda p: len(p[0]), reverse = True):
            if name in class_name:
                cls = cand
                break
    
    if cls is None:
        raise ValueError('Unknown {} class for pretrained model {}\n  Candidates : {}'.format(
            _name, pretrained_name, tuple(_cls.keys())
        ))

    return cls.from_pretrained(pretrained_name = pretrained_name, ** kwargs)


get_pretrained_transformer      = partial(__get_pretrained, _transformers, 'transformer')
get_pretrained_transformer_encoder  = partial(__get_pretrained, _encoders, 'encoder')
get_pretrained_transformer_decoder  = partial(__get_pretrained, _decoders, 'decoder')

#__load()

