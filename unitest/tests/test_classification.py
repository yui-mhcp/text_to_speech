
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

from unitest import Test, assert_function, assert_equal

def test_classifier(model_name, dataset):
    from models.classification import BaseClassifier
    
    if isinstance(dataset, str):
        from datasets import get_dataset
        
        dataset = get_dataset(dataset)
    
    valid = dataset
    if isinstance(dataset, dict): valid = dataset.get('valid', dataset['test'])
    elif isinstance(dataset, (list, tuple)) and len(dataset) == 2: valid = dataset[1]
    
    model = BaseClassifier(nom = model_name)
    
    assert_equal((28, 28, 1), model.input_size)
    
    for i, data in enumerate(valid):
        if i >= 5: break
        image, label = data['image'], data['label']
        
        assert_function(model.predict, data['image'])
        assert_equal(data['label'], lambda image: model.predict(image)[0][0], data['image'])
    
@Test(sequential = True, model_dependant = 'mnist_classifier')
def test_base_classifier():
    test_classifier('mnist_classifier', 'mnist')
