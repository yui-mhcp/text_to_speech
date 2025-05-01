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

from threading import Event

class AsyncResult:
    """ This class is inspired from the `AsyncResult` used in `multiprocessing.Pool`s """
    def __init__(self, callback = None):
        self._callback  = callback
        
        self._event     = Event()
        self._result    = None
    
    @property
    def ready(self):
        return self._event.is_set()
    
    def __call__(self, result):
        self._result    = result
        self._success   = not isinstance(result, Exception)
        if self._callback is not None:
            self._callback(self._result)
        self._event.set()
    
    def wait(self, timeout = None):
        self._event.wait(timeout)
    
    def get(self, timeout = None):
        self.wait(timeout)
        if not self.ready:
            raise TimeoutError
        elif self._success:
            return self._result
        else:
            raise self._result
    
