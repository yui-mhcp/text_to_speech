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

from threading import Thread, RLock

from utils.thread_utils.producer import Producer, Item, update_item
from utils.thread_utils.consumer import Consumer

logger = logging.getLogger(__name__)

class Grouper(Consumer):
    """
        A `Grouper` is a regular `Consumer` excepts that it outputs 1 item for (at least) 1 input item by grouping multiple items with the same `group.id`
        
        In practice, it should be set after a `Splitter` instance that properly initializes the `group` field
    """
    def __init__(self, consumer = None, * args, nested_group = True, suffix = None, ** kwargs):
        super().__init__(consumer = self.group, * args, ** kwargs)
        
        self.suffix = suffix
        self.nested_group   = nested_group
        
        self.mutex_group    = RLock()
        self.group_items    = {}

    def group(self, group_id, * args, ** kwargs):
        if isinstance(group_id, (list, tuple)): return [self.group(g_id) for g_id in group_id]
        
        with self.mutex_group:
            items = sorted(self.group_items.pop(group_id), key = lambda it: it.group.index)
        if items[0].group.total == -1: items.pop()
        logger.debug('[GROUPER {}] Grouping ID {} ({} items)'.format(self.name, group_id, len(items)))
        
        nested_group = self.nested_group and all(isinstance(it.data, dict) for it in items)
        
        group = Item(
            data = {} if nested_group else [], args = args, kwargs = kwargs, id = group_id,
            priority    = min([it.priority for it in items]),
            callback    = items[0].callback
        )
        for i, item_i in enumerate(items):
            if item_i.group.index != i:
                raise RuntimeError('Item #{} has number {} !\nAll items :\n{}'.format(
                    i, item_i.group.index, '\n'.join([str(it) for it in items])
                ))
            
            if nested_group:
                for k, v in item_i.data.items():
                    if self.suffix and not k.endswith(self.suffix): k += self.suffix
                    group.data.setdefault(k, []).append(v)
            else:
                group.data.append(item_i.data)
        
        return group
    
    def append(self, item, * args, ** kwargs):
        if not isinstance(item, Item) or item.group is None:
            raise RuntimeError('Grouper got an invalid Item or no group : {}'.format(item))
        
        group = item.group
        with self.mutex_group:
            logger.debug('[GROUPER {}] Item {} / {} for ID {} added !'.format(
                self.name, group.index, group.total, group.id
            ))
            self.group_items.setdefault(group.id, []).append(item)
            
            n = len(self.group_items[group.id])
            if n == group.total:
                super().append(group.id, * args, priority = item.priority, ** kwargs)
            elif n > group.total and group.total != -1:
                raise RuntimeError('Item id {} has {} items but only {} are expected'.format(
                    group.id, n, group.total
                ))
        
        return item
