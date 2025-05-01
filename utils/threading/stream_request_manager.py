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

import queue
import logging

from threading import Lock, Event
from multiprocessing import Pipe

from .process import run_in_thread

logger = logging.getLogger(__name__)

def StreamRequestManager():
    """
        This function returns a parent and a child streaming manager, to control streaming generation between processes.
        
        In the normal use case, it can simply act like a `Queue` by forwarding results from the child to the parent process, while handling multiple streams in parallel, and correctly mapping each result (from child process) to the expected stream (in the parent process).
        
        In a more complex usage, the parent process can send messages to the child process, and vice-versa, to control the stream (e.g., stop / finalize it).
    """
    parent_pipe, child_pipe = Pipe()
    return ParentRequestManager(parent_pipe), ChildRequestManager(child_pipe)

class RequestManager:
    def __init__(self, pipe):
        self.pipe   = pipe
        
        self.built  = False
        self.mutex  = None
        self.mutex_pipe = None
    
    def build(self):
        if self.built: return
        
        self.built = True
        
        self.mutex  = Lock()
        self.mutex_pipe = Lock()
    
    def send(self, message):
        """ Send `message` in the pipe (expected a `dict` with "id", "type" and "content" entries) """
        with self.mutex_pipe:
            self.pipe.send(message)

    def send_action(self, request_id, action):
        self.send({'id' : request_id, 'type' : 'action', 'content' : action})

    def send_result(self, request_id, output):
        self.send({'id' : request_id, 'type' : 'output', 'content' : output})

    def send_status(self, request_id, status):
        self.send({'id' : request_id, 'type' : 'status', 'content' : status})


class ParentRequestManager(RequestManager):
    """
        This class handles stream results comming from the child process, and maps them to the correct buffer. 
    """
    def __init__(self, pipe):
        super().__init__(pipe)
        
        self._stopped   = False
        self._requests  = {}
        self._request_idx   = 0
        
        self.build()
        self.handle_message()

    def stop(self):
        self._stopped = True
    
    def init_request(self, request_id = None):
        with self.mutex:
            idx = self._request_idx
            self._request_idx += 1
        
            if request_id is None: request_id = idx

            self._requests[request_id] = buffer = queue.Queue()
        
        self.send_action(request_id, 'init')

        return request_id, buffer
    
    def abort_request(self, request_id):
        self.send_action(request_id, 'stop')
    
    def finalize_request(self, request_id):
        self.send_action(request_id, 'finalize')

    @run_in_thread(daemon = True)
    def handle_message(self):
        while not self._stopped:
            msg = self.pipe.recv()
            if msg['id'] in self._requests:
                self._requests[msg['id']].put(msg)
                if msg['type'] == 'status' and msg['content'] == 'finished':
                    self._requests.pop(msg['id'])
            else:
                logger.error('The request id {} is not active, and should therefore not receive any message, but got {}.'.format(msg['id'], msg))
    
class ChildRequestManager(RequestManager):
    def __init__(self, pipe):
        super().__init__(pipe)
        
        self._stopped   = set()
        self._requests  = {}
    
    def build(self):
        if self.built: return

        super().build()
        
        self.check_for_message()
    
    def __call__(self, item):
        """ Send a stream event from the child to the parent process """
        if isinstance(item, tuple):
            item = {'id' : item[0], 'type' : 'output', 'content' : item[1]}
        
        if item['id'] not in self._requests:
            logger.error('The request {} seems to not have been initialized ! Make sure to send a status "init" message (in the parent process) before starting it.'.format(item['id']))
        
        self.send(item)
        
        if item['type'] == 'status' and item['content'] == 'finished':
            self.pop(item['id'])
        
        return self.is_active(item['id'])
    
    def is_active(self, request_id):
        return not self.is_stopped(request_id)
    
    def is_stopped(self, request_id):
        with self.mutex:
            return request_id in self._stopped
    
    def is_finalized(self, request_id):
        with self.mutex:
            return request_id in self._requests and not self._requests[request_id].is_set()
    
    def wait_finalize(self, request_id):
        """ Wait until the request is stopped or finalized, and return `True` if the request was finalized, and `False` if it was stopped """
        self._requests[request_id].wait()
        return not self.is_stopped(request_id)
    
    def pop(self, request_id):
        with self.mutex:
            self._requests.pop(request_id)
            return request_id not in self._stopped
            
    @run_in_thread(daemon = True)
    def check_for_message(self):
        while True:
            msg = self.pipe.recv()
            if msg['type'] == 'action':
                with self.mutex:
                    if msg['content'] == 'init':
                        self._requests[msg['id']] = Event()
                        logger.info('Request {} initialized'.format(msg['id']))
                    elif msg['id'] not in self._requests:
                        logger.error('The request id {} has been finalized, and should not receive new messages, but got {}'.format(msg['id'], msg))
                    elif msg['content'] == 'stop':
                        self._stopped.add(msg['id'])
                        self._requests[msg['id']].set()
                        logger.info('Request {} is requested to stop'.format(msg['id']))
                    elif msg['content'] == 'finalize':
                        self._requests[msg['id']].set()
                        logger.info('Request {} is finalized'.format(msg['id']))

