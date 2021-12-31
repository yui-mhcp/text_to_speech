import os
import time
import logging
import requests

from queue import Queue
from threading import Thread, Lock

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
            raise ValueError("You must provide bot token !")
        self._maybe_get_chat_id()
        if self.chat_id is None:
            logging.warning("You must give a chat_id or send a message to the bot to allow it to get chat id\n \
            It must be a message and not a command !)")
    
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
