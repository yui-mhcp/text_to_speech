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
import copy
import enum
import queue
import logging
import functools
import multiprocessing

from typing import Optional, Any
from threading import Thread, RLock
from dataclasses import dataclass, field

from utils.stream_utils import create_iterator
from utils.generic_utils import get_enum_item, get_args, signature_to_str
from utils.wrapper_utils import partial

logger = logging.getLogger(__name__)

class StoppedException(Exception):
    pass

class Event(enum.IntEnum):
    START   = 0
    APPEND  = 1
    ITEM_APPEND = 1
    ITEM    = 2
    ITEM_PRODUCED   = 2
    STOP    = 3

@dataclass(order = True)
class Item:
    data    : Any   = field(compare = False, repr = False)
    priority    : Optional[Any] = -1
    index       : Optional[int] = -1
    
    stop    : Optional[bool]    = field(compare = False, default = False)
    
    args    : Optional[tuple] = field(compare = False, default = (), repr = False)
    kwargs  : Optional[dict]  = field(compare = False, default_factory = dict, repr = False)
    
    callback    : Optional[callable]    = field(compare = False, default = None, repr = False)

ITEM_EVENTS = (Event.ITEM, Event.APPEND)

STOP_ITEM   = Item(data = None, stop = True)
_propagate_item_fields = ('index', 'priority', 'group', 'args', 'kwargs', 'metadata')

def update_item(item, clone = True, ** kwargs):
    if clone: item = copy.copy(item)
    for k, v in kwargs.items(): setattr(item, k, v)
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
                 start_listeners    = None,
                 append_listeners   = None,
                 stop_listeners     = None,
                 
                 daemon     = False,
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
        Thread.__init__(self, name = _get_thread_name(generator, name), daemon = daemon)

        self.generator  = generator
        self.description    = generator.__doc__ if not description and hasattr(generator, '__doc__') else description
        self.run_main_thread    = run_main_thread
        self.raise_if_error     = raise_if_error
        self.stop_no_more_listeners = stop_no_more_listeners

        self._listeners     = {}
        
        self.mutex_infos    = RLock()
        self._stopped   = False
        self._finished  = False
        self._size  = 0
        
        for (event, listeners) in zip(
            [Event.START, Event.APPEND, Event.STOP],
            [start_listeners, append_listeners, stop_listeners]):
            
            if listeners is None: continue
            if not isinstance(listeners, (list, tuple)): listeners = [listeners]
            for l in listeners: self.add_listener(l, event = event)
        
        if consumers is not None:
            if not isinstance(consumers, (list, tuple)): consumers = [consumers]
            for c in consumers: self.add_consumer(c, start = True, link_stop = True)
    
    @property
    def size(self):
        with self.mutex_infos: return self._size

    @property
    def stopped(self):
        with self.mutex_infos: return self._stopped
    
    @property
    def finished(self):
        with self.mutex_infos: return self._finished

    @property
    def start_listeners(self):
        return self._listeners.get(Event.START, [])
    
    @property
    def item_listeners(self):
        return self._listeners.get(Event.ITEM, [])

    @property
    def stop_listeners(self):
        return self._listeners.get(Event.STOP, [])
    
    @property
    def str_status(self):
        if self.run_main_thread:    return '/'
        elif self.is_alive():   return 'alive'
        elif self.stopped:      return 'stopped'
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
        self._iterator  = self.generator() if callable(self.generator) else create_iterator(self.generator)
        return self._iterator
    
    def __len__(self):
        return len(self.generator) if self.generator is not self and hasattr(self.generator, '__len__') else self.size
    
    def __str__(self):
        des = self.node_text
        des += '# Listeners : start : {} - append : {} - items  : {} - stop  : {}\n'.format(
            len(self.start_listeners),
            len(self.append_listeners),
            len(self.item_listeners),
            len(self.stop_listeners)
        )
        return des
    
    def run(self):
        """ Starts the producer, iterates on the `generator` then stops the thread """
        self.on_start()
        for item in self:
            if self.stopped:
                if hasattr(self._iterator, 'close'): self._iterator.close()
                break
            self.on_item_produced(item)
        self.on_stop()

    def start(self):
        if self.run_main_thread: self.run()
        else: super().start()
        return self

    def stop(self):
        with self.mutex_infos:
            self._stopped = True
        
        if isinstance(self.generator, (queue.Queue, multiprocessing.queues.Queue)):
            self.generator.put(STOP_ITEM)

    def add_listener(self,
                     listener,
                     * args,
                     event  = Event.ITEM,
                     name   = None,
                     pass_item  = False,
                     ** kwargs
                    ):
        """
            Add a `listener` (callable) called at the given `event`
            If the event is `item`, the first argument received is the produced item
            args / kwargs are given when called
            
            /!\ Listeners are executed in the Producer's thread so make sure to use `Consumer`'s running on separated threads to ensure a correct parallelization
        """
        if not callable(listener):
            raise ValueError('`listener` must be a callable ! Got type {}'.format(type(listener)))

        event = get_enum_item(event, Event)
        
        if isinstance(listener, Producer) and event == Event.ITEM:
            logger.debug('[LISTENER {}] consumer added !'.format(self.name))
        else:
            logger.debug('[LISTENER {}] listener added on `{}` event !'.format(
                self.name, Event(event).name
            ))
        
        infos = {
            'name'      : _get_listener_name(listener) if name is None else name,
            'stopped'   : False,
            'pass_item' : pass_item
        }
        
        if event in ITEM_EVENTS:
            if not get_args(listener):
                raise RuntimeError('The listener {} must accept at least 1 positional argument !\n  Signature : {}'.format(listener, signature_to_str(listener)))
            if isinstance(listener, Producer): infos['consumer_class'] = listener
            if args or kwargs:
                listener = functools.wraps(listener)(
                    lambda item, * a, ** kw: listener(item, * (args + a), ** {** kwargs, ** kw})
                )
        elif args or kwargs:
            listener = partial(listener, * args, ** kwargs)
        
        self._listeners.setdefault(event, []).append((listener, infos))
    
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
        from utils.thread_utils import Consumer
        
        if not isinstance(consumer, Consumer): consumer = Consumer(consumer, * args, ** kwargs)
        
        self.add_listener(consumer, event = Event.ITEM, pass_item = True)
        if link_stop:
            self.add_listener(
                consumer.stop, event = Event.STOP, name = 'stop {}'.format(consumer.name)
            )
            consumer.add_listener(self.stop, event = Event.STOP, name = 'stop {}'.format(self.name))
        if start and not consumer.is_alive(): consumer.start()

        return consumer
    
    def join(self, * args, recursive = False, wakeup_timeout = 0.2, ** kwargs):
        logger.debug('[JOIN {}] Waiting...'.format(self.name))
        if not self.run_main_thread:
            try:
                while super().is_alive():
                    super().join(timeout = kwargs.get('timeout', wakeup_timeout))
                    if 'timeout' in kwargs: break
            except KeyboardInterrupt:
                logger.info('Thread stopped while being joined !')
                self.stop()
        
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
        with self.mutex_infos: self._stopped, self._finished = True, True
        self.stop()
        for l, _ in self.stop_listeners: l()

    def on_item_produced(self, item):
        """ Function called when a new item is generated """
        with self.mutex_infos:
            idx = self._size
            self._size += 1
        
        if not isinstance(item, Item):
            item = Item(data = item, index = idx)
        
        logger.debug('[ITEM PRODUCED {}] {}'.format(self.name, item))
        if item.stop:
            self.stop()
            return
        
        if item.callback is not None: item.callback()
        
        to_remove = []
        consumers = self.item_listeners
        for i, (consumer, infos) in enumerate(consumers):
            if infos.get('stopped', False): continue
            try:
                consumer(item) if infos.get('pass_item', False) else consumer(
                    item.data, * item.args, ** item.kwargs
                )
            except Exception as e:
                print(e)
                stopped = isinstance(e, StoppedException)
                logger.error('[CONSUMER {} {}] : consumer {}{}'.format(
                    'STOPPED' if stopped else 'ERROR', self.name, infos['name'],
                    '' if stopped else '\n{}'.format(e)
                ))
                to_remove.append(i)
                if self.raise_if_error and not stopped:
                    self.stop()
                    raise e
        
        if self.stop_no_more_listeners and consumers and len(to_remove) == len(consumers):
            logger.debug('[STATUS {}] no more consumers, stopping the thread'.format(self.name))
            self.stop()
        
        if to_remove:
            for idx in reversed(to_remove): consumers.pop(idx)
    
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
    
