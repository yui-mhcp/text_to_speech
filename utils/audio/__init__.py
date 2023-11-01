
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

from utils.audio.audio_search import AudioSearch, SearchResult
from utils.audio.audio_annotation import *
from utils.audio.audio_processing import *
from utils.audio.audio_augmentation import *

from utils.audio.audio_io import play_audio, display_audio, load_audio, load_mel
from utils.audio.audio_io import read_audio, write_audio, resample_file, resample_audio
from utils.audio.audio_io import _load_fn, _write_fn

from utils.audio.mkv_utils import process_mkv, parse_subtitles

from utils.audio.stft import *

_audio_formats = tuple(set(list(_load_fn.keys()) + list(_write_fn.keys())))

from utils.file_utils import load_data, dump_data

load_data.dispatch(read_audio, tuple(_load_fn.keys()))
dump_data.dispatch(write_audio, tuple(_write_fn.keys()))
