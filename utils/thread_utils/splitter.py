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

from threading import Thread

from utils.thread_utils.producer import Producer, Item, Group, update_item
from utils.thread_utils.consumer import Consumer
from utils.thread_utils.threaded_dict import ThreadedDict

logger = logging.getLogger(__name__)

class Splitter(Consumer):
    """
        Splitter is a classical Consumer except that the output of the `consumer` function is expected to be a splitted version of the input
        It means that the output should be either :
            - a tuple (group_id, group_parts)
            - an Item with `group_id = res.group (or res.id)` and `group_parts = res.items`
        
        The Splitter will automatically creates a specific `Item` instance for each part, propagate the priority / args / kwargs and properly set the `group` attribute to a `Group` instance
        Note that `group_id` should be a regular identifier (and not a `Group` instance)
        
        Items in `group_parts` can be instance of `Item` and have their own `id` field which will not be influenced by the new `group` field. If parts do not have a specific `id`, they will be identified by their group (i.e. `group.id` and `group.index`)
    """
    def update_res_item(self, item, res):
        def item_generator():
            total = len(results) if hasattr(results, '__len__') else -1
            for i, res_i in enumerate(results):
                if total == -1 and res_i is None: total = i+1
                g = Group(id = group_id, index = i, total = total)
                yield update_item(item, data = res_i, group = g, id = None, clone = True)
                
        if isinstance(res, Item):
            group_id, results = res.group if res.group is not None else res.id, res.items
        elif isinstance(res, tuple):
            group_id, results = res
        else:
            raise ValueError('Unexpected result to be splitted\n  Expected : tuple (group_id, group_parts) or Item with `group` (or `id`) field and `items`\n  Got : {}'.format(res))
        
        return Item(data = None, items = item_generator())
