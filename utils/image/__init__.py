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

from .video import *
from .image_io import *
from .bounding_box import *
from .image_normalization import get_image_normalization_fn
from .image_processing import resize_image, pad_image, get_output_size

_image_formats  = ('gif', 'png', 'jpeg', 'jpg')

from ..file_utils import load_data, dump_data

load_data.dispatch(load_image, _image_formats)
dump_data.dispatch(save_image, _image_formats)
