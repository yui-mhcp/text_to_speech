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

from .ops_builder import build_op, build_custom_op

shuffle = build_op('keras.random.shuffle', 'random.shuffle', disable_np = True)
uniform = build_op('keras.random.uniform', 'random.uniform', disable_np = True)
normal  = build_op('keras.random.normal', 'random.normal', disable_np = True)
randint = build_op(
    'keras.random.randint', 'random.uniform', tf_kwargs = {'dtype' : 'int32'}, disable_np = True
)