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

from utils.text.document_parser import parse_document

@parse_document.dispatch(('txt', 'md'))
def parse_txt(filename, encoding = 'utf-8', paragraph_sep = '\n\n', ** kwargs):
    with open(filename, 'r', encoding = encoding) as f:
        text = f.read()
    
    return [{'text' : p} for p in text.split(paragraph_sep)]

