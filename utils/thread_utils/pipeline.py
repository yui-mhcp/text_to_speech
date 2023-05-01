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

import os
import logging
import pandas as pd

from queue import PriorityQueue
from threading import Thread, RLock

from utils.file_utils import load_data, dump_data
from utils.thread_utils.producer import Producer, Event, Item, Group, update_item
from utils.thread_utils.consumer import Consumer
from utils.thread_utils.splitter import Splitter
from utils.thread_utils.grouper import Grouper

logger = logging.getLogger(__name__)

def get_item_id(item, use_group = False, data_as_default = True, id_key = 'id'):
    if use_group and item.group is not None:
        return item.group
    elif item.id is not None:
        return item.id
    elif data_as_default:
        data_id = None
        if isinstance(item.data, str):
            data_id = item.data
        elif isinstance(item.data, (dict, pd.Series)) and id_key in item.data:
            data_id = item.data[id_key]
        
        if data_id is not None: item = update_item(item, id = data_id, clone = False)
        return data_id
    else:
        return None

class Pipeline(Consumer):
    """
        A `Pipeline` is a regular `Consumer` wrapping multiple consumers inside of it, allowing some useful features such as saving and tracking duplicates
        
        The `filename` argument allows to save a mapping `{item_id : final_result` where
        - item_id is either `item.id` or `item.data` (if it is a string)
        - final_result is the final pipeline's output for the given item_id
        The `expected_keys` tells which keys are required to use a restored version of item_id. It means that if `item_id` is in the database but some keys are missing, it will be re-processed
        The `do_not_save_keys` allows to skip some keys in the output file. Note that those keys are still returned in the pipeline, they are simply not saved in the database.
        **Important** : as a result, if some keys are both in `expected_keys` and `do_not_save_keys`, the database will never be used to not re-process items as some keys will always be missing. 
        
        `track_items` allows to track duplicates and only process each ID only once while returning all requested occurences of it. `keep_order` tells whether to return the duplicates in the append's order or in the production order. See the example below for more information
        
        Example on tracking :
        if keep_order: # Example 1
            Append ID 1 : in_pipeline[i]['indexes'] = [0] -> add to process
            Append ID 2 : in_pipeline[2]['indexes'] = [1] -> add to process
            Append ID 1 : in_pipeline[i]['indexes'] = [0, 2] -> /
            Output ID 1 : _results[0] = output 1, _results[2] = output 1
            Output ID 2 : _results[1] = output 2
            
            self.results = [output 1, output 2, output 1]
        else: # Example 2
            Append ID 1 : in_pipeline[i]['indexes'] = [0] -> add to process
            Append ID 2 : in_pipeline[2]['indexes'] = [1] -> add to process
            Append ID 1 : in_pipeline[i]['indexes'] = [0, 2] -> /
            Output ID 1 : _results[0] = output 1
                          _results[1] = output 1
            Output ID 2 : _results[2] = output 2
            
            self.results = [output 1, output 1, output 2]
        
        In the 1st example, the items in `self.results` corresponds the append's order while in the 2nd example, once an item is produced, it will be added as mny times as it has been requested.
        Note that in both cases, the 2nd request for ID 1 does not add the item in the process pipeline (i.e. it will simply waits that the 1st occurence goes out).
        Note that these examples are simplified : in practice, each replica stores its `Item` instance to also keep tracks of its original fields (such as `priority` and `group`)
        
        **Warning** if `track_items = True`, items with an ID already in the pipeline will not be re-processed, assuming that the `data` field is the same for both as the ID is the same.
        If some items can have the same ID but different data : either set `track_items` to False or modify the identifier to differenciate them.
    """
    def __init__(self,
                 tasks,
                 * args,
                 
                 filename   = None,
                 id_key     = 'id',
                 save_every = -1,
                 save_kwargs    = None,
                 as_list    = False,
                 expected_keys  = None,
                 save_keys      = None,
                 do_not_save_keys   = None,
                 
                 track_items    = None,
                 keep_order     = None,
                 
                 name   = None,
                 
                 ** kwargs
                ):
        def _add_tracker(i, consumer, splitted = False, grouped = False,
                         is_pipe = False, prev = None):
            if isinstance(consumer, Pipeline):
                for j in range(len(consumer.consumers)):
                    splitted, grouped = _add_tracker(
                        i, consumer.consumers[j], splitted, grouped, is_pipe = True, prev = prev
                    )
                    prev = consumer.consumers[j]
                return splitted, grouped
            
            default_kwargs = {
                'cons' : consumer, 'pass_item' : True, 'splitted' : splitted, 'grouped' : grouped
            }
            
            if i > 0 or is_pipe:
                if not isinstance(consumer, Grouper):
                    consumer.add_listener(
                        self._move_in_pipeline, on = 'append', ** default_kwargs
                    )
                elif prev is not None:
                    prev.add_listener(
                        self._move_in_pipeline, on = 'item', ** {** default_kwargs, 'cons' : None}
                    )
            
            if isinstance(consumer, Splitter):
                consumer.add_listener(
                    self._add_splitted, on = 'item', _first = True, ** default_kwargs
                )
            
            if isinstance(consumer, Splitter): return (True, grouped)
            if isinstance(consumer, Grouper): return (splitted, True)
            return (splitted, grouped)
        
        
        self.consumers  = []
        cons = Producer.build_consumer(tasks[0], ** kwargs)
        self.consumers.append(cons)
        for task in tasks[1:]:
            if not isinstance(task, dict) or 'consumer' not in task: task = {'consumer' : task}
            task.update({'link_stop' : True, 'start' : False})
            cons = cons.add_consumer(** {** kwargs, ** task})
            self.consumers.append(cons)
        
        if name is None: name = '-'.join([cons.name for cons in self.consumers])
        kwargs.setdefault('description', 'Pipeline with {} tasks'.format(len(tasks)))
        kwargs.update({'batch_size' : 1, 'allow_multithread' : False})
        super().__init__(consumer = lambda x: x, * args, name = name, ** kwargs)
        if filename is not None: track_items = True
        
        self.last.add_listener(
            self._append_result, on = 'item', pass_item = True, name = 'pipeline output'
        )
        self.last.add_listener(
            lambda: super(Pipeline, self).stop(), on = 'stop', name = 'stop {}'.format(self.name)
        )
        
        self.filename   = filename
        self.id_key     = id_key
        self.save_every = save_every
        self.save_kwargs    = {'indent' : 4} if filename and filename.endswith('json') and save_kwargs is None else (save_kwargs if save_kwargs is not None else {})
        self.as_list    = as_list
        self.expected_keys  = expected_keys if expected_keys is not None else []
        self.save_keys      = save_keys if save_keys is not None else []
        self.do_not_save_keys   = do_not_save_keys if do_not_save_keys is not None else []
        
        if track_items is None: track_items = True if filename is not None else False
        self.track_items    = track_items
        self.keep_order     = keep_order
        
        self.__saved    = False
        self.__database = None
        self.__in_pipeline  = None
        self.mutex_track    = RLock() if self.filename and self.track_items else None
        self.mutex_pipe = None
        self.mutex_db   = None
        
        if filename is not None:
            if not isinstance(self.expected_keys, (list, tuple)):
                self.expected_keys = [self.expected_keys]
            if not isinstance(self.save_keys, (list, tuple)):
                self.save_keys = [self.save_keys]
            if not isinstance(self.do_not_save_keys, (list, tuple)):
                self.do_not_save_keys = [self.do_not_save_keys]

            if not os.path.splitext(filename)[1]:
                self.filename += '.json'
            
            os.makedirs(os.path.dirname(self.filename), exist_ok = True)
            
            self.expected_keys      = set(self.expected_keys)
            self.do_not_save_keys   = set(self.do_not_save_keys)
            
            self.mutex_db   = RLock()
            self.__database = self.load_database()
            
            self.add_listener(self.save_database, on = 'stop')
            
        if track_items:
            if keep_order is None:
                keep_order = False if isinstance(self.buffer, PriorityQueue) else True
                logger.warning('`keep_order` has been set to {} by default as the buffer is{} a PriorityQueue.\nTo avoid this message, set explicitely `keep_order`'.format(
                    keep_order, ' not' if keep_order else ''
                ))
            
            self.mutex_pipe     = RLock()
            self.__in_pipeline  = {}
            
            splitted, grouped = False, False
            for i in range(len(self.consumers)):
                splitted, grouped = _add_tracker(
                    i, self.consumers[i], splitted, grouped, prev = self.consumers[i-1]
                )
            
            if splitted != grouped:
                raise NotImplementedError('Splitting and grouping must have the same value but got {} and {} : it indicates you either have put a `Splitter` without `Grouper` or the opposit case.\n  To solve this, either perform the split *before* the `Pipeline` or group *after* it\n  Pipeline : {}\n'.format(
                    splitted, grouped, '\n'.join(['- {} : {}'.format(c.name, type(c)) for c in self.consumers])
                ))
        
        logger.debug('[PIPELINE CREATED {}] {} consumers successfully initialized !'.format(
            self.name, len(self.consumers)
        ))
    
    def load_database(self):
        db = load_data(self.filename, default = {})
        
        if isinstance(db, pd.DataFrame): db = db.to_dict('records')
        if isinstance(db, list):
            self.as_list = True
            db = {row[self.id_key] : row for row in db}
        
        return db
    
    def save_database(self):
        with self.mutex_db:
            if not self.__saved:
                self.__saved = True
                data = self.__database if not self.as_list else [
                    {** v, self.id_key : k} for k, v in self.__database.items()
                ]
                dump_data(filename = self.filename, data = data, ** self.save_kwargs)
    
    def _get_from_database(self, item, * args):
        if not self.filename:
            if len(args) > 0: return args[0]
            raise RuntimeError('`filename` is None, cannot use the database feature')
        
        if not isinstance(item, Item): item = Item(data = item)
        item_id = get_item_id(item, use_group = False, data_as_default = True, id_key = self.id_key)
        return self.__database.get(item_id, * args)
    
    def _load_from_database(self, item):
        if not self.filename:
            raise RuntimeError('`filename` is None, cannot use the database feature')
        
        item_id = get_item_id(
            item, use_group = False, data_as_default = True, id_key = self.id_key
        )
        if item_id is None:
            logger.warning('[PIPELINE DB ADD {}] Unable to determine the ID for {}'.format(
                self.name, item
            ))
            return True
        
        contains = False
        with self.mutex_db:
            if item_id in self.__database:
                contains, result = True, self.__database[item_id]
        
        if contains:
            if item.kwargs.get('overwrite', False):
                item.kwargs = {** item.kwargs, 'overwritten_data' : result}
                return True
                
            if not self.expected_keys or all(k in result for k in self.expected_keys):
                self._set_result(
                    update_item(item, data = self.__database[item_id], clone = False)
                )
                logger.debug('[PIPELINE DB GET {}] Restored from database ID `{}`'.format(
                    self.name, item_id
                ))
                return False
            
            logger.debug('[PIPELINE DB GET {}] Some keys are missing from the saved data for ID `{}`'.format(self.name, item_id))
        
        return True
    
    def _add_in_database(self, item):
        if not self.filename:
            raise RuntimeError('`filename` is None, cannot use the database feature')
        
        item_id = get_item_id(item, use_group = False, data_as_default = False)
        if item_id is None:
            logger.warning('[PIPELINE DB ADD {}] Unable to determine the ID for {}'.format(
                self.name, item
            ))
            return
        
        data_to_save = item.data
        if self.save_keys:
            data_to_save = {
                k : v for k, v in data_to_save.items() if k in self.save_keys
            }
        elif self.do_not_save_keys:
            data_to_save = {
                k : v for k, v in data_to_save.items() if k not in self.do_not_save_keys
            }
        
        with self.mutex_db:
            self.__saved = False
            self.__database[item_id] = data_to_save
            if (
                (self.save_every > 0 and len(self.__database) % self.save_every == 0)
                or (self.save_every == -1 and all(cons.empty() for cons in self.consumers))):
                self.save_database()
        
        logger.debug('[PIPELINE DB ADD {}] Add ID `{}` in database'.format(self.name, item_id))
    
    def _add_in_pipeline(self, item, cons, use_group = False, ** kwargs):
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')
        
        item_id = get_item_id(
            item, use_group = use_group, data_as_default = True, id_key = self.id_key
        )
        if item_id is None:
            logger.warning('[PIPELINE NOT TRACKING {}] Item does not have a valid identifier, cannot track it : {}'.format(self.name, item))
            return True
        
        with self.mutex_pipe:
            if item_id in self.in_pipeline:
                self._add_duplicate(item, ** kwargs)
                return False

            self.in_pipeline[item_id] = {
                'cons' : cons, 'item' : item, 'original_items' : [item]
            }
        logger.debug('[PIPELINE ADD {}] {}'.format(self.name, item_id))
        return True

    def _add_duplicate(self, item, ** kwargs):
        def _update_priority(cons, item, prio):
            if cons is not None:
                cons.update_priority(item, prio, keep_best = True)
            else:
                item.priority = min(item.priority, prio)
        
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')
        
        item_id = get_item_id(item, use_group = False, data_as_default = False)
        with self.mutex_pipe:
            if item_id not in self.in_pipeline:
                raise RuntimeError('Try to add duplicate for ID {} but it is not in the pipeline !'.format(item_id))
            
            infos = self.in_pipeline[item_id]
            infos['original_items'].append(item)
            
            if infos['item'] is not None:
                _update_priority(infos['cons'], infos['item'], item.priority)
            else:
                assert infos.get('parts', None) is not None
                for part in infos['parts']:
                    part_infos = self.in_pipeline[part]
                    _update_priority(part_infos['cons'], part_infos['item'], item.priority)
        logger.debug('[PIPELINE ADD {}] Add duplicate for ID `{}`'.format(self.name, item_id))

    def _add_splitted(self, item, cons, ** kwargs):
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')
        
        item_id = get_item_id(item, use_group = True)
        logger.debug('[PIPELINE ADD PART {}] {}'.format(self.name, item_id))
        
        with self.mutex_pipe:
            if item_id in self.in_pipeline:
                raise RuntimeError('Item part {} is already in pipeline which should not be possible !')
            
            self._add_in_pipeline(item, cons = cons, use_group = True, ** kwargs)
            
            original_id = item.group.id
            if original_id not in self.in_pipeline:
                raise RuntimeError('Original ID {} for part with ID {} is not in pipeline !'.format(
                    original_id, item_id
                ))
            
            item = self.in_pipeline[original_id]['item']
            if item is not None:
                self.in_pipeline[original_id].update({
                    'item' : None, 'args' : item.args, 'kwargs' : item.kwargs
                })
            self.in_pipeline[original_id].setdefault('parts', []).append(item_id)
    
    def _get_from_pipeline(self, item, use_group = False):
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')
        item_id = get_item_id(
            item, use_group = use_group, data_as_default = True, id_key = self.id_key
        )
        with self.mutex_pipe: return self.in_pipeline.get(item_id, None)
    
    def _move_in_pipeline(self, item, cons, splitted = False, grouped = False, ** kwargs):
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')
        
        item_id = get_item_id(
            item, use_group = True if splitted and not grouped else False, data_as_default = False
        )
        with self.mutex_pipe:
            if item_id not in self.in_pipeline:
                raise RuntimeError('Try to move ID {} but it is not in the pipeline !'.format(item_id))
            if 'kwargs' in self.in_pipeline[item_id]:
                item.args   = self.in_pipeline[item_id].pop('args')
                item.kwargs = self.in_pipeline[item_id].pop('kwargs')
            self.in_pipeline[item_id].update({'item' : item, 'cons' : cons})
        
        logger.debug('[PIPELINE MOVE {}] Move ID {} to {}'.format(
            self.name, item_id, cons.name if cons is not None else 'None (waiting for group)'
        ))
    
    def _remove_from_pipeline(self, item, splitted = False, grouped = False, ** kwargs):
        if not self.track_items: raise RuntimeError('Items in pipeline are not tracked !')

        item_id = get_item_id(item, use_group = False, data_as_default = False)
        with self.mutex_pipe:
            if item_id not in self.in_pipeline:
                raise RuntimeError('Try to remove ID {} but it is not in the pipeline !'.format(item_id))
            
            infos = self.in_pipeline.pop(item_id)
            
            removed = 1 + len(infos.get('parts', []))
            for part_id in infos.get('parts', []):
                self.in_pipeline.pop(part_id)
        
        logger.debug('[PIPELINE REMOVE {}] {} item(s) removed for ID `{}` !'.format(
            self.name, removed, item_id
        ))
        return infos
    
    @property
    def in_pipeline(self):
        return self.__in_pipeline
    
    @property
    def database(self):
        return self.__database
    
    @property
    def first(self):
        return self.consumers[0]
    
    @property
    def last(self):
        return self.consumers[-1]
    
    @property
    def node_text(self):
        des = super().node_text
        if self.track_items:    des += "Tracking items\n"
        if self.filename:       des += "Saving to {}\n".format(self.filename)
        return des

    def __next__(self):
        with self.mutex_infos:
            if self._stop_index != -1 and self._current_index >= self._stop_index:
                raise StopIteration()

            idx = self._current_index
            self._current_index += 1

        logger.debug('[PIPELINE NEXT {}] Waiting for index {}'.format(self.name, idx))
        item = self._results[idx]
        if not self.keep_result: self._results.pop(idx)
        if item.stop: raise StopIteration()
        return item
    
    def get_consumer(self, name):
        for cons in self.consumers:
            if cons.name == name: return cons
        return None

    def _set_result(self, item):
        with self.mutex_infos:
            idx = self._next_index if not self.keep_order else item.index
            self._next_index += 1
        self._results[idx] = update_item(item, index = idx, clone = False)

        logger.debug('[PIPELINE RESULT {}] Set result for index {}'.format(self.name, idx))
        return item
        
    def _append_result(self, item, * args, ** kwargs):
        if not self.track_items:
            if self.filename is not None: self._add_in_database(item)
            return self._set_result(item)
        
        elif self.filename is None: # track_items = True
            infos = self._remove_from_pipeline(item)
        else: # track_items = True and filename is not None
            with self.mutex_track:
                self._add_in_database(item)
                
                infos = self._remove_from_pipeline(item)
        
        return [
            self._set_result(update_item(original, data = item.data, clone = True))
            for original in infos['original_items']
        ]
    
    def add_listener(self, * args, ** kwargs):
        if kwargs.get('on', None) != Event.MOVE_IN_PIPELINE:
            return super().add_listener(* args, ** kwargs)
        
        kwargs.pop('on')
        for i in range(len(self.consumers)):
            self.consumers[i].add_listener(
                * args, on = 'append', cons = self.consumers[i], ** kwargs
            )

    def _append_item(self, item):
        self.on_append(item)

        should_append = True
        if self.filename and self.track_items:
            with self.mutex_track:
                should_append = self._load_from_database(item)
                if should_append:
                    should_append = self._add_in_pipeline(item, cons = self.first)
        elif self.filename:
            should_append = self._load_from_database(item)
        elif self.track_items:
            should_append = self._add_in_pipeline(item, cons = self.first)

        if should_append:
            self.first.append(item)
        else:
            logger.debug('[PIPELINE WAIT {}] Item {} already in pipeline'.format(
                self.name, get_item_id(item)
            ))
        
        if self.run_main_thread:
            self.on_item_produced(self.__next__())
    
    def get(self, * args, ** kwargs):
        raise RuntimeError('This function should never be called')

    def get_batch(self, * args, ** kwargs):
        raise RuntimeError('This function should never be called')

    def consume_next(self, * args, ** kwargs):
        raise RuntimeError('This function should never be called')

    def consume(self, * args, ** kwargs):
        raise RuntimeError('This function should never be called')
    
    def start(self):
        for cons in self.consumers:
            if not cons.is_alive(): cons.start()
        super().start()
    
    def stop(self, * args, ** kwargs):
        self.first.stop(* args, ** kwargs)
    
    def stop_when_empty(self, * args, ** kwargs):
        for cons in self.consumers: cons.stop_when_empty(* args, ** kwargs)
        super().stop_when_empty(* args, ** kwargs)
    
    def join(self, * args, ** kwargs):
        for cons in self.consumers: cons.join(* args, ** kwargs)
        super().join(* args, ** kwargs)
    
    def wait(self, * args, ** kwargs):
        for cons in self.consumers: cons.wait(* args, ** kwargs)
        super().wait(* args, ** kwargs)

    def plot(self, graph = None, node_graph = None, node_id = 0,
             name = None, filename = None, view = False, ** kwargs):
        import graphviz as gv
        if graph is None:
            if name is None: name = filename if filename else self.name
            graph = gv.Digraph(name = name, filename = filename)
            graph.attr(compound = 'true')

        if node_graph is None: node_graph = graph
        
        pipe_graph_name = 'cluster_pipeline{}'.format(self.name)
        with graph.subgraph(name = pipe_graph_name) as sub_graph:
            sub_graph.attr(label = 'Pipeline {}'.format(self.name))
            
            graph, child_id, next_id = self.first.plot(
                graph = graph, node_graph = sub_graph, node_id = node_id,
                ** {** kwargs, 'view' : False}
            )
        
        res = super().plot(graph = graph, node_graph = node_graph, node_id = next_id, ** kwargs)
        
        if node_id == 0 and (view or filename is not None):
            basename, format = (os.path.splitext(filename)[0], filename.split('.')[-1]) if filename is not None else (None, 'pdf')
            graph.render(
                filename = basename, view = view, cleanup = True, format = format
            )
        return res
