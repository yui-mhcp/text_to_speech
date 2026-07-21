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

import logging

from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class Callback(ABC):
    # Declarative capabilities read by the generic `apply_callbacks` / `predict`
    # orchestration, so the core never has to know about concrete callback types :
    #   - `saves_to_disk`  : the callback persists file(s) -> it is skipped when
    #                        `apply_callbacks(..., save = False)`
    #   - `provides_entry` : the callback's `apply` return value is the canonical
    #                        stored "entry" -> captured by `apply_callbacks` and used
    #                        by `predict` to decide the default of `return_output`
    saves_to_disk   = False
    provides_entry  = False

    def __init__(self, name = None, cond = None, initializer = None, ** _):
        self.name   = name or self.__class__.__name__
        self.cond   = cond
        self.initializer    = initializer
        
        self.built  = False
    
    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)
    
    def build(self):
        self.built = True
    
    def __call__(self, infos, output, ** kwargs):
        if self.cond is not None and not self.cond(** output):
            return
        elif self.initializer:
            for k, fn in self.initializer.items():
                if k not in output: output[k] = fn(** output)
        
        if not self.built: self.build()

        return self.apply(infos, output, ** kwargs)
    
    @abstractmethod
    def apply(self, infos, output, ** kwargs):
        """ Apply the callback """

    def join(self):
        pass