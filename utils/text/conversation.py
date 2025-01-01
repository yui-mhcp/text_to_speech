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

import time
import uuid

from typing import List, Dict, Any
from dataclasses import dataclass, field

from utils import load_data, dump_data

@dataclass(order = True)
class Message:
    content : Any   = field(compare = False)
    lang    : str   = field(default = 'en', compare = False, repr = False)
    content_type    : str   = field(default = 'text', compare = False, repr = False)
    
    role    : str = field(default = 'user', compare = False)
    user    : str = field(default = None, compare = False)
    conv_id : Any = field(default_factory = uuid.uuid4, compare = False, repr = False)
    conv_name   : str   = field(default = None, compare = False, repr = False)
    
    time    : float = field(default_factory = time.time, repr = False)
    infos   : Dict[str, Any]    = field(default_factory = dict, compare = False, repr = False)
    
    @property
    def text(self):
        return self.content
    
    def __hash__(self):
        return hash((self.time, self.user, self.content))

    def __getitem__(self, key):
        if key == 'text': return self.text
        return self.__dict__[key] if hasattr(self, key) else self.infos[key]
    
    def __contains__(self, key):
        return key == 'text' or key in self.__dict__ or key in self.infos
    
    def filter(self, *, all_match = True, ** kwargs):
        fn = all if all_match else any
        return fn(getattr(self, k) == v for k, v in kwargs.items())
    
@dataclass(unsafe_hash = True)
class Conversation:
    messages    : List[Message] = field(default_factory = list, hash = False)
    id          : Any   = field(default_factory = uuid.uuid4, repr = False, hash = True)
    
    @property
    def last_conv(self):
        return self.filter(conv_id = self.last_conv_id)
    
    @property
    def last_conv_id(self):
        return self[-1].conv_id if self.messages else None

    @property
    def last_conv_name(self):
        return self[-1].conv_name if self.messages else None

    def __len__(self):
        return len(self.messages)
    
    def __contains__(self, user):
        return any(msg.user == user for msg in self.messages)
    
    def __getitem__(self, idx):
        if isinstance(idx, (int, slice)):   return self.messages[idx]
        elif isinstance(idx, str):          return [msg for msg in self.messages if msg.user == idx]
        else:   raise ValueError('Unsupported index : {}'.format(idx))
    
    def append(self, message, *, new_conv = None, ** kwargs):
        if new_conv is None: new_conv = kwargs.get('role', 'user') == 'user'
        
        if not new_conv and 'conv_id' not in kwargs:
            kwargs.update({'conv_id' : self.last_conv_id, 'conv_name' : self.last_conv_name})
            
        if not isinstance(message, Message):
            kwargs = {k : v for k, v in kwargs.items() if k in Message.__match_args__}
            message = Message(message, ** kwargs)
        self.messages.append(message)
    
    def filter(self, *, all_match = True, ** kwargs):
        if not self.messages: return []
        kwargs = {k : v for k, v in kwargs.items() if hasattr(self.messages[0], k)}
        return [msg for msg in self.messages if msg.filter(all_match = all_match, ** kwargs)]
    
    def save(self, filename):
        return dump_data(filename, self)
    
    @classmethod
    def load(cls, filename):
        conv = load_data(filename, default = {})
        if 'messages' in conv:
            conv['messages'] = [Message(** msg) for msg in conv['messages']]
        
        return cls(** conv)
