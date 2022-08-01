import logging
import numpy as np
import pandas as pd

from typing import Any
from queue import Queue
from collections import deque
from threading import Thread, RLock
from dataclasses import dataclass, field

@dataclass(order = True)
class Result:
    priority    : Any
    index       : int
    result      : Any   = field(compare = False)

class StoppedException(Exception):
    def __init__(self, * args, ** kwargs):
        super(StoppedException, self).__init__(* args, ** kwargs)

def _create_generator(generator):
    if isinstance(generator, (list, tuple, np.ndarray)):
        return lambda: iter(generator)
    elif isinstance(generator, pd.DataFrame):
        def _df_iterator():
            for idx, row in generator.iterrows():
                yield row
        return _df_iterator
    elif isinstance(generator, Queue):
        def _queue_iterator():
            while True:
                item = queue.get()
                if item is None: raise StopIteration()
                yield item
        return _queue_iterator
    else:
        raise ValueError('Unknown generator type ({}) : {}'.format(type(generator), generator))
    
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
                 stop_listeners     = None,
                 
                 run_main_thread    = False,
                 stop_no_more_listeners = True,
                 
                 name = None,
                 ** kwargs
                ):
        """
            Constructor for the `Producer`
            
            Arguments :
                - generator : the callable function that returns a `generator` (or `iterator`)
                
                - consumers : consumers to add
                - {start / stop}_listeners  : listeners to add on start / stop events
                
                - run_main_thread   : whether to run in a separate Thread or not
                - stop_no_more_listeners    : whether to call `self.stop` when the producer has lost all its consumers (note that if a producer (or consumer) has never had any consumer, this argument is ignored)
                - name      : the Thread's name
        """
        Thread.__init__(self, name = _get_thread_name(generator, name))
        
        self.generator  = generator if callable(generator) else _create_generator(generator)
        self.description    = self.generator.__doc__ if not description and hasattr(self.generator, '__doc__') else description
        self.run_main_thread    = run_main_thread
        self.stop_no_more_listeners = stop_no_more_listeners

        self.start_listeners    = []
        self.item_listeners     = []
        self.stop_listeners     = []
        
        self.mutex_infos    = RLock()
        self._stopped   = False
        self._finished  = False
        self.__size    = 0
        
        if start_listeners is not None:
            if not isinstance(start_listeners, (list, tuple)): start_listeners = [start_listeners]
            for l in start_listeners: self.add_listener(l, on = 'start')
        
        if consumers is not None:
            if not isinstance(consumers, (list, tuple)): consumers = [consumers]
            for c in consumers: self.add_consumer(c, start = True, link_stop = True)
        
        if stop_listeners is not None:
            if not isinstance(stop_listeners, (list, tuple)): stop_listeners = [stop_listeners]
            for l in stop_listeners: self.add_listener(l, on = 'stop')
    
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
        if self.run_main_thread: return '/'
        elif self.is_alive(): return 'alive'
        elif self.finished: return 'finished'
        else: return 'not started'
    
    @property
    def node_text(self):
        des = "{}\n".format(self.name)
        if self.description: des += "{}\n".format(self.description)
        des += "Thread : {}\n".format(self.str_status)
        return des
        
    def __iter__(self):
        return self.generator()
    
    def __len__(self):
        return len(self.generator) if self.generator is not self and hasattr(self.generator, '__len__') else self.size
    
    def __str__(self):
        des = 'Producer {}:\n'.format(self.name)
        des += 'Thread alive : {}\n'.format(self.is_alive())
        des += 'Already produced {} items\n'.format(self.size)
        des += '# Listeners :\n- Start : {}\n- Items  : {}\n- Stop  : {}\n'.format(
            len(self.start_listeners), len(self.item_listeners), len(self.stop_listeners)
        )
        return des
    
    def add_listener(self, listener, * args, on = 'item', ** kwargs):
        """
            Add a `listener` (callable) called at the given (`on`) event
            If the event is `item`, the first argument received is the produced item
            args / kwargs are given when called
            
            /!\ Listeners are executed in the Producer's thread so make sure to use `Consumer`'s running on separated threads to ensure a correct parallelization
        """
        assert on in('item', 'start', 'stop')
        
        if not callable(listener):
            raise ValueError('`listener` must be a callable ! Got type {}'.format(type(listener)))
        
        if isinstance(listener, Producer) and on == 'item':
            logging.debug('[LISTENER {}] consumer added !'.format(self.name))
        else:
            logging.debug('[LISTENER {}] listener added on `{}` event !'.format(self.name, on))
        
        infos = {'name' : _get_listener_name(listener), 'stopped' : False}
        if on == 'item':
            if isinstance(listener, Producer): infos['consumer_class'] = listener
            self.item_listeners.append((
                lambda item, ** kw: listener(item, * args, ** {** kwargs, ** kw}), infos
            ))
        elif on == 'start':
            self.start_listeners.append((lambda: listener(* args, ** kwargs), infos))
        elif on == 'stop':
            self.stop_listeners.append((lambda: listener(* args, ** kwargs), infos))
    
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
        from utils.thread_utils.consumer import Consumer
        
        if not isinstance(consumer, Consumer):
            if not isinstance(consumer, list):
                if not isinstance(consumer, dict): consumer = {'consumer' : consumer}
                consumer    = Consumer(* args, ** {** kwargs, ** consumer})
            else:
                from utils.thread_utils.pipeline import Pipeline
                consumer    = Pipeline(consumer, * args, ** kwargs)
        
        self.add_listener(consumer, on = 'item')
        if link_stop:
            self.add_listener(consumer.stop, on = 'stop')
            consumer.add_listener(self.stop, on = 'stop')
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
    
    def stop(self):
        with self.mutex_infos:
            self._stopped = True
    
    def join(self, * args, recursive = False, ** kwargs):
        logging.debug('[{}JOIN {}]{}'.format(
            'RECURSIVE ' if recursive else '', self.name,
            ' {} consumers'.format(len(self.item_listeners)) if recursive else ''
        ))
        if not self.run_main_thread:
            super().join(* args, ** kwargs)
        
        if recursive:
            for l, infos in self.item_listeners:
                if 'consumer_class' not in infos: continue
                infos['consumer_class'].join(* args, recursive = True, ** kwargs)
        logging.debug('[JOIN {}] Joined !'.format(self.name))
    
    def on_start(self):
        """ Function called when starting the thread """
        logging.debug('[STATUS {}] Start'.format(self.name))
        for l, _ in self.start_listeners: l()

    def on_stop(self):
        """ Function called when stopping the thread """
        logging.debug('[STATUS {}] Stop'.format(self.name))
        self.stop()
        with self.mutex_infos: self._finished = True
        for l, _ in self.stop_listeners: l()

    def on_item_produced(self, item):
        """ Function called when a new item is generated """
        logging.debug('[ITEM PRODUCED {}]'.format(self.name))
        with self.mutex_infos:
            self.__size += 1
        
        for l, infos in self.item_listeners:
            if infos.get('stopped', False): continue
            try:
                l(item) if not isinstance(item, Result) else l(item.result, priority = item.priority)
            except StoppedException:
                logging.debug('[CONSUMER STOPPED {}] consumer {} stopped'.format(
                    self.name, infos['name']
                ))
                infos['stopped'] = True
        
        _stopped = [l for l, i in self.item_listeners if i.get('stopped', False)]
        if len(_stopped) > 0 and len(_stopped) == len(self.item_listeners) and self.stop_no_more_listeners:
            logging.debug('[STATUS {}] no more consumers, stopping the thread'.format(self.name))
            self.stop()
    
    def plot(self, filename = None, name = None, view = True, graph = None,
             node_graph = None, node_id = 0):
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
        
        str_id = str(node_id)
        
        plot_style = {
            'id'    : str_id,
            'shape' : "box" if type(self) == Producer else "ellipse",
            'label' : self.node_text.replace('\n', '\l').replace('\l', '\n', 1)
        }
        if node_graph is None: node_graph = graph
        
        node_graph.node(str_id, ** plot_style)
        
        _add_listeners_graph(self.start_listeners, 'Start listeners', 'on_start')

        next_id = node_id + 1
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

        if node_id == 0 and view:
            graph.view()
        
        return graph, str_id, next_id

