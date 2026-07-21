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

""" Shared fixtures for the `utils.image.bounding_box` test-suite.

Replaces the former ``_base.py`` (``TestBoxes`` / ``unittest`` + ``absl``) with
plain ``pytest`` fixtures. Kept ``numpy``-only at import time : the keras-free
(numpy) cases never pull a backend, while the ``tensor`` cases convert through
``utils.keras.ops`` (which forces a real backend) and are therefore marked
``keras`` via the shared ``backend`` fixture below (mirrors ``test_distances``).
"""

import numpy as np
import pytest

# reference image size (h, w) used to normalize relative <-> absolute boxes
IMAGE_H, IMAGE_W = 720, 1024

# the same 3 boxes expressed in both supported formats. Row 0 spans the whole
# image, rows 1/2 are arbitrary inner boxes.
_RELATIVE = {
    'xywh' : np.array([[0, 0, 1, 1], [0.25, 0.2, 0.1, 0.2], [0.5, 0.5, 0.5, 0.5]], dtype = 'float32'),
    'xyxy' : np.array([[0, 0, 1, 1], [0.25, 0.2, 0.35, 0.4], [0.5, 0.5, 1, 1]], dtype = 'float32'),
}


@pytest.fixture
def image_shape():
    """ Reference image size as ``(h, w)``. """
    return (IMAGE_H, IMAGE_W)

@pytest.fixture
def relative_boxes():
    """ ``dict`` of ``float32`` boxes in `[0, 1]`, keyed by format (`xywh` / `xyxy`). """
    return {fmt: boxes.copy() for fmt, boxes in _RELATIVE.items()}

@pytest.fixture
def absolute_boxes():
    """ ``dict`` of ``int32`` boxes in pixel space, keyed by format (`xywh` / `xyxy`). """
    factor = np.array([[IMAGE_W, IMAGE_H, IMAGE_W, IMAGE_H]])
    return {fmt: (boxes * factor).astype('int32') for fmt, boxes in _RELATIVE.items()}


# --- backend handling (numpy = keras-free, tensor = keras) --------------------------

_BACKENDS = [
    pytest.param('numpy',  id = 'numpy'),
    pytest.param('tensor', id = 'tensor', marks = pytest.mark.keras),
]

@pytest.fixture(params = _BACKENDS)
def backend(request):
    """ Parametrized backend : ``numpy`` (unmarked) or ``tensor`` (marked ``keras``). """
    return request.param
