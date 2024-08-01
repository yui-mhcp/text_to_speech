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
import glob
import importlib

from .parser_utils import *
from .parser import parse_document, first_page_to_image
from .html_parser import _wiki_cleaner

def __load():
    for module in glob.glob(__package__.replace('.', os.path.sep) + '/*_parser*'):
        if os.path.basename(module).startswith(('.', '_')): continue
        importlib.import_module(module.replace(os.path.sep, '.').replace('.py', ''))

__load()
