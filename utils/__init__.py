
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

from utils.distance import *
from utils.embeddings import *
from utils.file_utils import *
from utils.plot_utils import *
from utils.pandas_utils import *
from utils.thread_utils import *
from utils.stream_utils import *
from utils.generic_utils import *
from utils.sequence_utils import *
from utils.tensorflow_utils import *
from utils.comparison_utils import *
from utils.wrapper_utils import *

logger = logging.getLogger(__name__)

_timer, _time_logger, _logger_available = None, logger, None

def get_timer():
    global _timer, _time_logger, _logger_available
    if _timer is None:
        try:
            from loggers import timer, time_logger
            _timer, _time_logger, _logger_available = timer, time_logger, True
        except ImportError as e:
            logger.warning('The `loggers` module is not available : the time performance tracking are disabled')
            logging.Logger.timer = lambda * _, ** __: ContextManager()
            _timer, _logger_available = fake_wrapper, False
    
    return _timer, _time_logger, _logger_available
