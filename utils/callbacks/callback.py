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

import logging

from loggers import time_logger

logger = logging.getLogger(__name__)

class Callback:
    def __init__(self, name = None, key = None, cond = None, initializers = {}, ** _):
        if not name: name = self.__class__.__name__
        
        self.key    = key
        self.name   = name
        self.cond   = cond
        self.initializers   = initializers
        
        self.built  = False
    
    def __repr__(self):
        des = '<{}'.format(self.__class__.__name__)
        if self.key: des += ' key={}'.format(self.key)
        return des + '>'
    
    def build(self):
        self.built = True
    
    def __call__(self, infos, output, ** kwargs):
        if self.cond is not None and not self.cond(infos = infos, ** {** kwargs, ** output}):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('- Skipping {} (condition not met)'.format(self))
            
            if self.key and self.key not in infos:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug('- Callback {} modified {}'.format(self, self.key))

                infos[self.key] = output.get(self.key, None)
                return True
            return False
        
        if not self.built: self.build()
        
        if output:
            for k, fn in self.initializers.items():
                if k not in output:
                    with time_logger.timer('{} init-{}'.format(self.name, k)):
                        output[k] = fn(** output)
            
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('- Apply {}'.format(self))
        
        out = self.apply(infos = infos, output = output, ** kwargs)
        if self.key:
            try:
                if infos.get(self.key, None) == out: return False
            except ValueError:
                pass
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('- Callback {} modified {}'.format(self, self.key))
            infos[self.key] = out
            return True
        # no side-effect on `infos`
        return False
    
    def apply(self, infos, output, ** kwargs):
        raise NotImplementedError()

    def join(self):
        pass