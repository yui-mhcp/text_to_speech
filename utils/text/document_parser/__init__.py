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

import os
import glob

from utils.text.document_parser.parser_utils import infer_pages, split_paragraphs, clean_paragraphs
from utils.text.document_parser.parser import parse_document, first_page_to_image
from utils.text.document_parser.html_parser import _wiki_cleaner

def __load():
    for module in glob.glob(__package__.replace('.', os.path.sep) + '/*_parser.py'):
        __import__(module.replace(os.path.sep, '.')[:-3])

__load()
