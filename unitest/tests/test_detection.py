
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

from unitest import Test, assert_function, assert_model_output, assert_equal

_filename = os.path.join('unitest', '__datas', 'lena.jpg')

@Test(sequential = True, model_dependant = 'yolo_faces')
def test_yolo_faces():
    from utils.image import load_image
    from models.detection import YOLO
    
    model = YOLO(nom = 'yolo_faces')
    
    image = load_image(_filename, target_shape = model.input_size)
    
    assert_equal(model.get_input, image, _filename)
    
    model.detect(image)
    
    assert_model_output(model.detect, image, get_boxes = False, training = False)
    assert_function(model.detect, image, get_boxes = True)
    
    output = model.detect(image, get_boxes = False)[0]
    assert_function(model.decode_output, output)
    assert_equal(lambda: len(model.decode_output(output)), 2)
    assert_equal(lambda: len(model.decode_output(output, obj_threshold = 0.5)), 1)
    
    boxes = model.decode_output(output)
    
    assert_function(model.draw_prediction, image, boxes)
    assert_function(model.draw_prediction, image, boxes, as_mask = True)

