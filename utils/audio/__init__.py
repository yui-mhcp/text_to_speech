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

from .audio_search import AudioSearch, SearchResult
from .audio_annotation import *
from .audio_processing import *
from .audio_augmentation import *

from .audio_io import *
from .audio_io import _load_fn, _write_fn

from .mkv_utils import process_mkv, parse_subtitles

from .stft import *

_audio_formats = tuple(set(list(_load_fn.keys()) + list(_write_fn.keys())))

from utils.file_utils import load_data, dump_data

load_data.dispatch(read_audio, tuple(_load_fn.keys()))
dump_data.dispatch(write_audio, tuple(_write_fn.keys()))
