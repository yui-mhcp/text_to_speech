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

from .parser import register_parser

@register_parser('txt', kind = 'text')
def read_txt(filename, *, encoding = 'utf-8', ** kwargs):
    """ Return the raw content of `filename` """
    with open(filename, 'r', encoding = encoding) as f:
        return f.read()

@register_parser('txt')
def parse_txt(filename, *, sep = '\n\n', ** kwargs):
    """ Extract a list of paragraphs by splitting the raw text on `sep` """
    return [{'text' : para} for para in read_txt(filename, ** kwargs).split(sep)]
