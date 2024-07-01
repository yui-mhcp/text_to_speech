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

import os
import time
import logging
import requests

from queue import Queue
from threading import Thread, Lock

logger  = logging.getLogger(__name__)

API_URL = 'https://api.telegram.org/bot{token}/{method}'

def get_chat_id(token):
    res = requests.post(API_URL.format(token = token, method = 'getUpdates'))
    if not res: return None
    res = res.json()
    if not res.get('ok', False) or len(res['result']) == 0: return None
    return res['result'][-1]['message']['chat']['id']

class TelegramHandler(logging.Handler):
    def __init__(self, token = None, chat_id = None, concat_delay = 1, level = 0, ** kwargs):
        super().__init__(level = level)
        self.token    = token if token is not None else os.environ.get('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id
        self.concat_delay = concat_delay
        
        self.queue  = Queue()
        self._sending = False
        self.mutex = Lock()
        
        if token is None:
            raise ValueError("You must provide bot token (`TELEGRAM_BOT_TOKEN` env variable) !")
        self._maybe_get_chat_id()
        if self.chat_id is None:
            logger.warning("You must give a `chat_id` (`TELEGRAM_CHAT_ID` env variable) or send a message to the bot to allow it to get chat id\n Note that it must be a message and not a command")
    
    def _maybe_get_chat_id(self):
        if self.chat_id is not None: return
        self.chat_id = os.environ.get('TELEGRAM_CHAT_ID', get_chat_id(self.token))
    
    def send(self, force = False):
        if not force:
            if self.concat_delay > 0: time.sleep(self.concat_delay)
        
        messages = []
        while not self.queue.empty():
            messages.append(self.format(self.queue.get()))
        
        if messages:
            data = {
                'text' : '\n'.join(messages),
                'chat_id' : self.chat_id,
                'disable_notification' : True,
                'disable_web_page_preview' : True
            }
            requests.post(API_URL.format(token = self.token, method = 'sendMessage'), json = data)
        
        with self.mutex:
            self._sending = False
    
    def emit(self, record):
        self.queue.put(record)
        self._maybe_get_chat_id()
        if self.chat_id is None: return
        
        with self.mutex:
            if self._sending: return
            self._sending = True
        Thread(target = self.send).start()
