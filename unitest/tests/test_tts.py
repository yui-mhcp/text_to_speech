
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

from unitest import Test, assert_function, assert_model_output, assert_equal

_pretrained_tacotron = 'pretrained_tacotron2'

_text_en = 'Hello World ! This is a small text to test the Tacotron 2 architecture in English.'
_text_fr = 'Bonjour à tous ! Ceci est un petit texte de test pour tester l\'architecture Tacotron 2.'

models_to_test = [_pretrained_tacotron, 'tacotron2_siwis', 'sv2tts_siwis']

def test_tacotron(model_name):
    from models import get_pretrained
    from models.tts import tts, SV2TTSTacotron2
    from models.model_utils import is_model_name
    from utils.text import default_english_encoder
    
    model = get_pretrained(model_name)
    model.tts_model.set_deterministic(True)
    
    assert_equal(148,   model.vocab_size)
    assert_equal(0,     model.blank_token_idx)
    assert_equal(80,    model.n_mel_channels)
    assert_equal(22050, model.audio_rate)
    
    if model.lang == 'en':
        assert_equal(default_english_encoder, model.text_encoder)
    
    assert_function(model.encode_text, _text_en)
    assert_function(model.encode_text, _text_fr)
    
    # These 2 lines are required in order to compile the model
    # because the result can differ between the 1st call (before compilation) and next calls (once compiled)
    model.predict([_text_en, _text_fr], save = False)
    
    if not isinstance(model, SV2TTSTacotron2):
        assert_model_output(model.infer, _text_en)
        assert_model_output(model.infer, _text_fr)
        assert_model_output(model.infer, model.encode_text(_text_en))
        assert_model_output(model.infer, model.encode_text(_text_fr))
    
    assert_model_output(model.predict, [_text_en, _text_fr], save = False)


    
for name in models_to_test:
    Test(
        lambda: test_tacotron(name),
        sequential  = True,
        model_dependant = name,
        name    = 'test_{}'.format(name)
    )
