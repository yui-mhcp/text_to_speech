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

import time
import logging
import requests
import numpy as np

from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

class HTTPScreenMirror:
    """ This class allows to stream based on the `http screen mirror` mobile app """
    def __init__(self, url, min_time = 1. / 16.):
        self.url    = url
        self.prefix = HTTPScreenMirror.get_prefix(url)
        self.min_time   = min_time
        self.last_time  = 0
    
    def __str__(self):
        return 'HTTP Screen Mirror ({})'.format(self.url)
    
    def read(self):
        wait = self.min_time - (time.time() - self.last_time)
        if wait > 0: time.sleep(wait)
        
        try:
            img = requests.get(
                '{}/{}{}.jpg'.format(self.url, self.prefix, int(time.time() * 1000))
            )
        except requests.ConnectionError as e:
            logger.warning('Server connection has been closed !')
            return False, None
        except Exception as e:
            logger.warning('Exception while reading frame : {}'.format(e))
            return False, None
        finally:
            self.last_time = time.time()
        
        if not img.content: return False, None
        return np.array(Image.open(BytesIO(img.content)))[..., ::-1]
    
    def release(self):
        pass
    
    @staticmethod
    def get_prefix(url):
        try:
            res = requests.get(url)
        except requests.ConnectionError:
            logger.error('Unable to connect to the given url : {}'.format(url))
            return None
        except Exception as e:
            raise e
        
        return res.content.decode().split('\n')[-9].split("'")[1]
        