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

from .ops_builder import build_op

__all__ = ['cholesky', 'det', 'eig', 'eigh', 'inv', 'lstsq', 'lu_factor', 'qr', 'solve', 'solve_triangular']

globals().update({k : build_op(k, disable_np = True) for k in __all__})