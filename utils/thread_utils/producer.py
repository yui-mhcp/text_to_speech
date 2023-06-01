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

import copy
import enum
import time
import logging
import numpy as np
import pandas as pd

from typing import Optional, Any
from queue import Queue
from collections import deque
from threading import Thread, Lock, RLock
from dataclasses import dataclass, field

from utils.generic_utils import get_enum_item, create_iterator

logger = logging.getLogger(__name__)
_group_mutex    = Lock()

class StoppedException(Exception):
    def __init__(self, * args, ** kwargs):
        super(StoppedException, self).__init__(* args, ** kwargs)

class Event(enum.IntEnum):
    START   = 0
    APPEND  = 1
    ITEM_APPEND = 1
    ITEM    = 2
    ITEM_PRODUCED   = 2
    STOP    = 3
    MOVE    = 4
    MOVE_IN_PIPELINE    = 4

@dataclass(order = True)
class Group:
    id  : Any
    index   : int
    total   : int

    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Group) or self.id is None or other.id is None: return False
        return self.id == other.id and (self.index == other.index or -1 in (self.index, other.index))

@dataclass(order = True)
class Item:
    data    : Any   = field(compare = False, repr = False)
    priority    : Optional[Any] = -1
    index       : Optional[int] = -1
    
    id      : Optional[Any] = field(compare = False, default = None)
    group   : Optional[Group]   = None
    
    stop    : Optional[bool]    = field(compare = False, default = False)
    items   : Optional[list] = field(compare = False, default = None, repr = False)
    
    args    : Optional[tuple] = field(compare = False, default = (), repr = False)
    kwargs  : Optional[dict]  = field(compare = False, default_factory = dict, repr = False)
    
    callback    : Optional[callable]    = field(compare = False, default = None, repr = False)
    metadata    : Optional[dict]  = field(compare = False, default_factory = dict, repr = False)

ITEM_EVENTS = (Event.ITEM, Event.APPEND)

STOP_ITEM   = Item(data = None, stop = True)
_propagate_item_fields = ('index', 'priority', 'group', 'args', 'kwargs', 'metadata')

def update_item(item, clone = True, ** kwargs):
    if 'data' in kwargs and isinstance(kwargs['data'], Item):
        for field in _propagate_item_fields:
            kwargs.setdefault(field, getattr(item, field))
        item = kwargs.pop('data')
        return update_item(item, ** kwargs)
    
    if clone: item = copy.copy(item)
    for k, v in kwargs.items():
        setattr(item, k, v)
    return item


def _get_thread_name(generator, name):
    if name is not None: return name
    if hasattr(generator, 'name'): return generator.name
    elif hasattr(generator, '__name__'): return generator.__name__
    return None

def _get_listener_name(listener):
    if hasattr(listener, 'name'): return listener.name
    elif hasattr(listener, '__name__'): return listener.__name__
    else: return listener.__class__.__name__

class Producer(Thread):
    """
        Thread that iterates on a `generator` in a Thread
        It does not store any result but has `listeners` that are called after each generation
    """
    def __init__(self,
                 generator,
                 * args,
                 
                 description    = None,
                 
                 consumers      = None,
                 append_listeners   = None,
                 start_listeners    = None,
                 stop_listeners     = None,
                 
                 run_main_thread    = False,
                 
                 raise_if_error     = True,
                 stop_no_more_listeners = True,
                 
                 name = None,
                 ** kwargs
                ):
        """
            Constructor for the `Producer`
            
            Arguments :
                - generator : the callable function that returns a `generator` (or `iterator`)
                    It can also be a `queue.Queue`, a `pd.DataFrame` or any type supporting iter(...)
                
                - description   : description for the `print()` and `plot()` methods
                
                - consumers : consumers to add
                - {start / append / stop}_listeners  : listeners to add on the specific event
                
                - run_main_thread   : whether to run in a separate Thread or not
                - stop_no_more_listeners    : whether to call `self.stop` when the producer has lost all its consumers (note that if a producer (or consumer) has never had any consumer, this argument is ignored)
                - name      : the Thread's name
            
            Difference between a `listener` and a `consumer` :
                Listeners are simple callables that are called in the class' thread.
                A`consumer` is a `Producer`-subclass (automatically created if needed) which can therefore run in a separate thread.  
        """
        Thread.__init__(self, name = _get_thread_name(generator, name))
        
        self.generator  = generator if callable(generator) else create_iterator(generator)
        self.description    = self.generator.__doc__ if not description and hasattr(self.generator, '__doc__') else description
        self.run_main_thread    = run_main_thread
        self.raise_if_error     = raise_if_error
        self.stop_no_more_listeners = stop_no_more_listeners

        self._listeners     = {}
        
        self.mutex_infos    = RLock()
        self._stopped   = False
        self._finished  = False
        self.__size    = 0
        
        for (event, listeners) in zip(
            [Event.START, Event.APPEND, Event.STOP],
            [start_listeners, append_listeners, stop_listeners]):
            
            if listeners is None: continue
            if not isinstance(listeners, (list, tuple)): listeners = [listeners]
            for l in listeners: self.add_listener(l, on = event)
        
        if consumers is not None:
            if not isinstance(consumers, (list, tuple)): consumers = [consumers]
            for c in consumers: self.add_consumer(c, start = True, link_stop = True)
    
    @property
    def start_listeners(self):
        return self._listeners.get(Event.START, [])
    
    @property
    def stop_listeners(self):
        return self._listeners.get(Event.STOP, [])
    
    @property
    def item_listeners(self):
        return self._listeners.get(Event.ITEM, [])
    
    @property
    def size(self):
        with self.mutex_infos: return self.__size
    
    @property
    def stopped(self):
        with self.mutex_infos: return self._stopped

    @property
    def finished(self):
        with self.mutex_infos: return self._finished
    
    @property
    def str_status(self):
        if self.run_main_thread:    return '/'
        elif self.is_alive():   return 'alive'
        elif self.finished:     return 'finished'
        else:   return 'not started'
    
    @property
    def node_text(self):
        des = "{} {}\n".format(self.__class__.__name__, self.name)
        if self.description: des += "{}\n".format(self.description)
        des += "Thread status : {}\n".format(self.str_status)
        des += "# items produced : {}\n".format(self.size)
        return des
        
    def __iter__(self):
        return self.generator() if callable(self.generator) else self.generator
    
    def __len__(self):
        return len(self.generator) if self.generator is not self and hasattr(self.generator, '__len__') else self.size
    
    def __str__(self):
        des = self.node_text
        des += '# Listeners :\n- Start : {}\n- Append : {}\n- Items  : {}\n- Stop  : {}\n'.format(
            len(self.start_listeners),
            len(self.append_listeners),
            len(self.item_listeners),
            len(self.stop_listeners)
        )
        return des
    
    def add_listener(self, listener, * args, on = 'item', pass_item = False, _first = False,
                     name = None, remove_when_stopped = False, ** kwargs):
        """
            Add a `listener` (callable) called at the given (`on`) event
            If the event is `item`, the first argument received is the produced item
            args / kwargs are given when called
            
            /!\ Listeners are executed in the Producer's thread so make sure to use `Consumer`'s running on separated threads to ensure a correct parallelization
        """
        event = get_enum_item(on, Event)
        
        if not callable(listener):
            raise ValueError('`listener` must be a callable ! Got type {}'.format(type(listener)))
        
        if isinstance(listener, Producer) and event == Event.ITEM:
            logger.debug('[LISTENER {}] consumer added !'.format(self.name))
        else:
            logger.debug('[LISTENER {}] listener added on `{}` event !'.format(self.name, on))
        
        infos = {
            'name'      : _get_listener_name(listener) if name is None else name,
            'stopped'   : False,
            'pass_item' : pass_item,
            'remove_when_stopped'   : remove_when_stopped
        }
        
        if event in ITEM_EVENTS:
            if isinstance(listener, Producer): infos['consumer_class'] = listener
            fn = lambda item, ** kw: listener(item, * args, ** {** kwargs, ** kw})
        else:
            fn = lambda: listener(* args, ** kwargs)
        
        if not _first:
            self._listeners.setdefault(event, []).append((fn, infos))
        else:
            self._listeners.setdefault(event, []).insert(0, (fn, infos))
    
    def add_consumer(self, consumer, * args, start = True, link_stop = False, ** kwargs):
        """
            Add a `Consumer` (possibly creates it)
            
            Arguments :
                - consumer  : the `callable` or `Consumer` instance that will consume the produced items
                - stateful  : whether to use a `StatefulConsumer` or a Consumer
                - start     : whether to start the Consumer's thread or not
                - link_stop : if True, it will call the Consumer's `stop` when the producer stops
                - args / kwargs : passed to the Consumer's constructor (if called)
            Return : the `Consumer` instance
        """
        consumer = Producer.build_consumer(consumer, * args, ** kwargs)
        
        self.add_listener(consumer, on = Event.ITEM, pass_item = True)
        if link_stop:
            self.add_listener(consumer.stop, on = Event.STOP, name = 'stop {}'.format(consumer.name))
            consumer.add_listener(self.stop, on = Event.STOP, name = 'stop {}'.format(self.name))
        if start and not consumer.is_alive(): consumer.start()

        return consumer
    
    def run(self):
        """ Start the producer, iterates on the `generator` then stops the thread """
        self.on_start()
        for item in self:
            if self.stopped: break
            self.on_item_produced(item)
        self.on_stop()
    
    def start(self):
        if self.run_main_thread: self.run()
        else: super().start()
        return self
    
    def stop(self):
        with self.mutex_infos:
            self._stopped = True
    
    def join(self, * args, recursive = False, ** kwargs):
        logger.debug('[JOIN {}] Waiting...'.format(self.name))
        if not self.run_main_thread:
            super().join(* args, ** kwargs)
        
        if recursive:
            logger.debug('[RECURSIVE JOIN {}] JOIN {} consumers'.format(
                self.name, len(self.item_listeners)
            ))
            for l, infos in self.item_listeners:
                if 'consumer_class' not in infos: continue
                infos['consumer_class'].join(* args, recursive = True, ** kwargs)
        logger.debug('[JOIN {}] Joined !'.format(self.name))
    
    def on_start(self):
        """ Function called when starting the thread """
        logger.debug('[STATUS {}] Start'.format(self.name))
        for l, _ in self.start_listeners: l()

    def on_stop(self):
        """ Function called when stopping the thread """
        logger.debug('[STATUS {}] Stop'.format(self.name))
        self.stop()
        with self.mutex_infos: self._finished = True
        for l, _ in self.stop_listeners: l()

    def on_item_produced(self, item):
        """ Function called when a new item is generated """
        with self.mutex_infos:
            idx = self.__size
            self.__size += 1
        
        if not isinstance(item, Item):
            item = Item(data = item, index = idx)
        
        logger.debug('[ITEM PRODUCED {}] {}'.format(self.name, item))
        if item.stop:
            self.stop()
            return
        
        if item.items is not None:
            logger.debug('[MULTI ITEMS {}] {} items produced'.format(
                self.name, len(item.items) if hasattr(item.items, '__len__') else '?'
            ))
        
        items = [item] if item.items is None else item.items
        for it in items:
            if it.callback is not None and it.callback[0] is self:
                it.callback[1](item.data)
            
            for l, infos in self.item_listeners:
                if infos.get('stopped', False): continue
                try:
                    l(it) if infos.get('pass_item', False) else l(it.data, * it.args, ** it.kwargs)
                except StoppedException:
                    logger.debug('[CONSUMER STOPPED {}] consumer {} stopped'.format(
                        self.name, infos['name']
                    ))
                    infos['stopped'] = True
                except Exception as e:
                    logger.error('[CONSUMER EXCEPTION {}] : consumer {}\n{}'.format(
                        self.name, infos['name'], e
                    ))
                    if self.raise_if_error:
                        raise e
                    else:
                        infos['stopped'] = True
        
        n_stopped = 0
        to_remove = []
        for idx, (l, infos) in enumerate(self.item_listeners):
            if not infos.get('stopped', False): continue
            if infos.get('remove_when_stopped', False): to_remove.append(idx)
            n_stopped += 1
        
        if n_stopped > 0 and n_stopped == len(self.item_listeners) and self.stop_no_more_listeners:
            logger.debug('[STATUS {}] no more consumers, stopping the thread'.format(self.name))
            self.stop()
        
        if to_remove:
            for idx in reversed(to_remove):
                self.item_listeners.pop(idx)
    
    def plot(self, filename = None, name = None, view = True, graph = None,
             node_graph = None, node_id = 0, node_name = None, str_id = None):
        """ Builds a `graphviz.Digraph` representing the producer-consumer pipeline """
        def _add_listeners_graph(listeners, label, edge_label):
            if len(listeners) == 0: return False
            g_name = 'cluster_{}{}'.format(str_id, edge_label)
            
            with graph.subgraph(name = g_name) as sub_graph:
                sub_graph.attr(label = label)

                for l, infos in listeners:
                    sub_graph.node(str(l), label = infos['name'], shape = 'circle')
            graph.edge(str_id, str(l), label = edge_label, lhead = g_name)
            return True
        
        import graphviz as gv
        if graph is None:
            if name is None: name = filename if filename else 'Graph'
            graph = gv.Digraph(name = name, filename = filename)
            graph.attr(compound = 'true')
        
        next_id = node_id
        if str_id is None:
            str_id  = str(node_id)
            next_id += 1

            plot_style = {
                'id'    : str_id,
                'shape' : "box" if type(self) == Producer else "ellipse",
                'label' : self.node_text.replace('\n', '\l').replace('\l', '\n', 1)
            }
            if node_graph is None: node_graph = graph

            node_graph.node(str_id, ** plot_style)
        
        _add_listeners_graph(self.start_listeners, 'Start listeners', 'on_start')

        if len(self.item_listeners) > 0:
            cons_graph_name = 'cluster_{}consumers'.format(str_id)
            with graph.subgraph(name = cons_graph_name) as sub_graph:
                sub_graph.attr(label = 'Consumers')

                for l, infos in self.item_listeners:
                    if 'consumer_class' in infos:
                        graph, child_id, next_id = infos['consumer_class'].plot(
                            graph = graph, node_graph = sub_graph, node_id = next_id, filename = None
                        )
                    else:
                        child_id = str(l)
                        sub_graph.node(child_id, label = infos['name'], shape = 'circle')
                graph.edge(str_id, child_id, lhead = cons_graph_name)
        
        _add_listeners_graph(self.stop_listeners, 'Stop listeners', 'on_stop')

        if node_id == 0 and (view or filename is not None):
            basename, format = (os.path.splitext(filename)[0], filename.split('.')[-1]) if filename is not None else (None, 'pdf')
            graph.render(
                filename = basename, view = view, cleanup = True, format = format
            )
        
        return graph, str_id, next_id
    
    @staticmethod
    def build_consumer(consumer, * args, ** kwargs):
        if isinstance(consumer, Producer): return consumer
        if not isinstance(consumer, dict):
            key = 'tasks' if isinstance(consumer, (list, tuple)) else 'consumer'
            consumer = {key : consumer}
        
        if 'tasks' in consumer:
            from utils.thread_utils.pipeline import Pipeline
            consumer    = Pipeline(* args, ** {** kwargs, ** consumer})
        elif consumer['consumer'] == 'grouper':
            from utils.thread_utils.grouper import Grouper
            consumer    = Grouper(* args, ** {** kwargs, ** consumer})
        elif consumer.pop('splitter', False):
            from utils.thread_utils.splitter import Splitter
            consumer    = Splitter(* args, ** {** kwargs, ** consumer})
        else:
            from utils.thread_utils.consumer import Consumer
            consumer    = Consumer(* args, ** {** kwargs, ** consumer})
    
        return consumer