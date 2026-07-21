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

""" Pure-``pytest`` tests for :mod:`utils.image.bounding_box.converter`.

Plain ``test_*`` functions + ``@pytest.mark.parametrize`` (no ``CustomTestCase`` /
``absl.parameterized``). Shared box fixtures come from the local ``conftest.py``,
tolerant comparisons through the ``asserts`` fixture.

Backend handling mirrors ``test_distances`` : ``numpy`` inputs stay keras-free
(``utils.keras.ops`` dispatches numpy), while ``tensor`` inputs go through
``ops.convert_to_tensor`` (which forces a real backend) and are marked ``keras``
by the shared ``backend`` fixture.
"""

import numpy as np
import pytest

from utils.keras import ops
from utils.image.bounding_box import convert_box_format

_FORMATS = ('xywh', 'xyxy')

def _to_tensor(x):
    return ops.convert_to_tensor(x)


# --- round-trip conversions (value, `as_list`, normalization) ----------------------

@pytest.mark.parametrize('target', _FORMATS)
@pytest.mark.parametrize('source', _FORMATS)
def test_convert_box_format(backend, asserts, relative_boxes, absolute_boxes, image_shape,
                            source, target):
    image_h, image_w = image_shape
    rel_in,  rel_out = relative_boxes[source], relative_boxes[target]
    abs_in,  abs_out = absolute_boxes[source], absolute_boxes[target]
    if backend == 'tensor':
        rel_in, rel_out = _to_tensor(rel_in), _to_tensor(rel_out)
        abs_in, abs_out = _to_tensor(abs_in), _to_tensor(abs_out)

    # relative -> relative, value and `as_list` (unstacked) variants
    asserts.assert_equal(rel_out, convert_box_format(rel_in, source = source, target = target))
    asserts.assert_equal(
        ops.unstack(rel_out, axis = -1, num = 4),
        convert_box_format(rel_in, source = source, target = target, as_list = True)
    )
    # absolute -> absolute
    asserts.assert_equal(abs_out, convert_box_format(abs_in, source = source, target = target))
    # relative -> absolute (normalization, needs the image size)
    asserts.assert_equal(abs_out, convert_box_format(
        rel_in, source = source, target = target,
        normalize_mode = 'absolute', image_h = image_h, image_w = image_w
    ))
    # absolute -> relative (int -> float, hence the rounding tolerance)
    asserts.assert_equal(rel_out, convert_box_format(
        abs_in, source = source, target = target,
        normalize_mode = 'relative', image_h = image_h, image_w = image_w
    ), max_err = 5e-4)


@pytest.mark.parametrize('fmt', _FORMATS)
def test_convert_box_format_identity_returns_same_instance(backend, relative_boxes, absolute_boxes, fmt):
    """ A same-format, non-normalizing conversion must return the very same object. """
    rel, abs_ = relative_boxes[fmt], absolute_boxes[fmt]
    if backend == 'tensor':
        rel, abs_ = _to_tensor(rel), _to_tensor(abs_)

    msg = 'The function should return the same instance when `source == target`'
    assert convert_box_format(rel, source = fmt, target = fmt) is rel, msg
    assert convert_box_format(rel, source = fmt, target = fmt, normalize_mode = 'relative') is rel, msg
    assert convert_box_format(abs_, source = fmt, target = fmt) is abs_, msg
    assert convert_box_format(abs_, source = fmt, target = fmt, normalize_mode = 'absolute') is abs_, msg


# --- `dezoom_factor` ---------------------------------------------------------------

@pytest.mark.parametrize('box,factor,expected', [
    # zoom-out keeps the box centered, scaling width / height by `factor`
    ([0., 0., 1., 1.],     0.5, [[0.25, 0.25, 0.5,  0.5 ]]),
    # dezoom is clipped to the [0, 1] range
    ([0., 0., 1., 1.],     2,   [[0,    0,    1,    1   ]]),
    ([0.25, 0.25, .5, .5], 2,   [[0,    0,    1,    1   ]]),
    ([0.5,  0.5,  .5, .5], 2,   [[0.25, 0.25, .75,  .75 ]]),
])
def test_convert_box_format_dezoom(asserts, box, factor, expected):
    asserts.assert_equal(
        np.array(expected, dtype = 'float64'),
        convert_box_format(box, source = 'xywh', dezoom_factor = factor)
    )
