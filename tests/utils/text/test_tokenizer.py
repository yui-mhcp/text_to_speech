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

""" Pure-``pytest`` tests for :mod:`utils.text.tokenizer`.

Ported from the original ``unittest`` ``TestTokenizer`` :
  - ``setUp`` -> a ``tokenizer`` fixture (a 150-symbol char-level english tokenizer) ;
  - ``absl.parameterized`` -> ``@pytest.mark.parametrize`` ;
  - the ``tensorflow`` graph test is gated behind ``@pytest.mark.tensorflow`` ;
  - the (very heavy, network) ``transformers`` round-trip test is gated behind
    ``pytest.importorskip('transformers')`` + ``@pytest.mark.slow``.

NOTE : the original ``_test_format`` (disabled via its leading underscore) is **not**
ported — it exercised a ``Tokenizer.format`` method that no longer exists, so it would
only raise ``AttributeError``. It is dropped rather than resurrected against a missing
API ; chat/templating is the job of ``encode_chat`` and should get its own tests.

Coverage was extended beyond the original port to harden the ``encode`` / ``decode``
hot path : ``decode`` / ``decode_ids`` (round-trip, padding skip & the consecutive-pad
early-break, ``remove_tokens``, batched / list-of-lists, float-logits ``argmax``),
``encode`` edge cases (``__call__`` alias, ``return_text``, independent ``add_sos`` /
``add_eos`` + idempotency, ``dict`` input, the cleaning path, out-of-vocab dropping,
empty input, invalid ``return_type``, ``tf`` / ``tensor`` return types), ``__getitem__``
/ ``__contains__`` and a ``get_config`` / ``save`` -> ``load_from_file`` round-trip.

Source bug found & fixed while writing these (``utils/text/tokenizer.py`` ``decode``) :
the ``remove_tokens`` branch set ``_skip = self.token_indexes``, a dict keyed by token
*string*, then tested the iterated integer ids against it (``token not in _skip``) — so
special tokens were never actually removed. Now ``_skip`` is a ``set`` of the id
*values*, unioned with the padding id (``remove_tokens`` also implies ``skip_padding``).
"""

import os

import numpy as np
import pytest

from utils.text import default_english_tokenizer, en_symbols, Tokenizer

_default_texts = [
    "Hello World !",
    "Bonjour à tous !",
    "1, 2, 3, 4, 5, 6, 7, 8, 9 et 10 !",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum.",
]


@pytest.fixture
def tokenizer():
    """ 150-symbol char-level english tokenizer with ``<s>`` / ``</s>`` at the vocab end. """
    return default_english_tokenizer(
        vocab_size      = 150,
        pad_token       = '_',
        sos_token       = '<s>',
        eos_token       = '</s>',
        use_sos_and_eos = True,
    )


# --- attributes / special tokens ---------------------------------------------------

def test_attributes(asserts, tokenizer):
    asserts.assert_equal(150, len(tokenizer))
    asserts.assert_equal(150, tokenizer.vocab_size)

    asserts.assert_equal(en_symbols, tokenizer.vocab[: len(en_symbols)])
    asserts.assert_equal(en_symbols, [tokenizer[i] for i in range(len(en_symbols))])

    asserts.assert_equal(en_symbols[0], tokenizer.blank_token)
    asserts.assert_equal(0,   tokenizer.blank_token_idx)
    asserts.assert_equal(148, tokenizer.sos_token_idx)
    asserts.assert_equal(149, tokenizer.eos_token_idx)


# --- encoding ----------------------------------------------------------------------

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_tokenize_keeps_special_tokens(asserts, tokenizer, text):
    asserts.assert_equal(
        ['<s>'] + list(text) + ['</s>'],
        tokenizer.tokenize(tokenizer.sos_token + text + tokenizer.eos_token, cleaned = True)
    )

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_encode(asserts, tokenizer, text):
    in_vocab = [tokenizer[c] for c in text if c in tokenizer]

    asserts.assert_equal(in_vocab, tokenizer.encode(text, cleaned = True, add_sos_and_eos = False))
    asserts.assert_equal(
        [tokenizer.sos_token_idx] + in_vocab + [tokenizer.eos_token_idx],
        tokenizer.encode(text, cleaned = True)
    )

def test_batched_encode(asserts, tokenizer):
    all_encoded = [tokenizer.encode(txt, cleaned = True) for txt in _default_texts]

    max_len = max(len(e) for e in all_encoded)
    all_encoded_padded = np.full(
        (len(all_encoded), max_len), tokenizer.blank_token_idx, dtype = np.int32
    )
    for i, enc in enumerate(all_encoded):
        all_encoded_padded[i, : len(enc)] = enc

    asserts.assert_equal(
        all_encoded, tokenizer.encode(_default_texts, cleaned = True, return_type = 'list')
    )
    asserts.assert_equal(
        all_encoded_padded, tokenizer.encode(_default_texts, cleaned = True, return_type = 'np')
    )


# --- `__getitem__` / `__contains__` ------------------------------------------------

def test_getitem(asserts, tokenizer):
    # int -> symbol and symbol -> int are inverse
    asserts.assert_equal('_', tokenizer[0])
    asserts.assert_equal(0,   tokenizer['_'])
    asserts.assert_equal('H', tokenizer[tokenizer['H']])

    # a multi-character string falls back to `encode` (no sos/eos) -> list of ids
    asserts.assert_equal([tokenizer['H'], tokenizer['i']], tokenizer['Hi'])

    # membership
    assert 'H' in tokenizer
    assert '中' not in tokenizer   # a CJK char, not in the english vocab

    # an unknown, non-string index raises
    with pytest.raises(KeyError):
        tokenizer[None]


# --- encoding (edge cases) ---------------------------------------------------------

def test_call_is_encode_alias(asserts, tokenizer):
    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), tokenizer(text))

def test_encode_return_text(asserts, tokenizer):
    text, encoded = tokenizer.encode('Hello', cleaned = True, return_text = True)
    asserts.assert_equal('Hello', text)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)

def test_encode_dict_input(asserts, tokenizer):
    asserts.assert_equal(
        tokenizer.encode('Hello', cleaned = True),
        tokenizer.encode({'text' : 'Hello'}, cleaned = True),
    )

def test_encode_add_sos_eos_independently(asserts, tokenizer):
    base = tokenizer.encode('Hello', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    sos, eos = tokenizer.sos_token_idx, tokenizer.eos_token_idx

    asserts.assert_equal(
        [sos] + base,
        tokenizer.encode('Hello', cleaned = True, add_sos = True, add_eos = False, return_type = 'list')
    )
    asserts.assert_equal(
        base + [eos],
        tokenizer.encode('Hello', cleaned = True, add_sos = False, add_eos = True, return_type = 'list')
    )
    asserts.assert_equal(
        [sos] + base + [eos],
        tokenizer.encode('Hello', cleaned = True, return_type = 'list')
    )

def test_encode_sos_eos_idempotent(asserts, tokenizer):
    """ Re-encoding a text that already carries ``<s>`` / ``</s>`` must not double them. """
    asserts.assert_equal(
        tokenizer.encode('Hello', cleaned = True, return_type = 'list'),
        tokenizer.encode(
            tokenizer.sos_token + 'Hello' + tokenizer.eos_token, cleaned = True, return_type = 'list'
        )
    )

def test_encode_out_of_vocab_dropped(asserts, tokenizer):
    """ Symbols outside the 150-token vocab (here digits) are silently dropped (no ``ukn``). """
    asserts.assert_equal(
        tokenizer.encode('Hi', cleaned = True, return_type = 'list'),
        tokenizer.encode('H123i', cleaned = True, return_type = 'list'),
    )

def test_encode_empty(asserts, tokenizer):
    asserts.assert_equal(
        [tokenizer.sos_token_idx, tokenizer.eos_token_idx],
        tokenizer.encode('', cleaned = True, return_type = 'list')
    )
    asserts.assert_equal(
        [], tokenizer.encode('', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    )

def test_encode_runs_cleaners(asserts, tokenizer):
    """ ``cleaned = False`` applies ``english_cleaners`` (lower-casing) before encoding. """
    asserts.assert_equal(
        tokenizer.encode('hello world', cleaned = True),
        tokenizer.encode('Hello World', cleaned = False),
    )
    asserts.assert_not_equal(
        tokenizer.encode('Hello World', cleaned = True),
        tokenizer.encode('Hello World', cleaned = False),
    )

def test_encode_invalid_return_type(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.encode('Hello', cleaned = True, return_type = 'banana')

@pytest.mark.tensorflow
def test_encode_return_type_tf(asserts, tokenizer):
    encoded = tokenizer.encode('Hello', cleaned = True, return_type = 'tf')
    asserts.assert_tf_tensor(encoded)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)

@pytest.mark.keras
def test_encode_return_type_tensor(asserts, tokenizer):
    encoded = tokenizer.encode('Hello', cleaned = True, return_type = 'tensor')
    asserts.assert_tensor(encoded)
    asserts.assert_equal(tokenizer.encode('Hello', cleaned = True), encoded)


# --- decoding ----------------------------------------------------------------------

@pytest.mark.parametrize('text', _default_texts, ids = ['hello', 'bonjour', 'numbers', 'lorem'])
def test_decode_roundtrip(asserts, tokenizer, text):
    # only in-vocab chars survive `encode` -> that's what `decode` can give back
    in_vocab = ''.join(c for c in text if c in tokenizer)
    encoded  = tokenizer.encode(text, cleaned = True)

    asserts.assert_equal(tokenizer.sos_token + in_vocab + tokenizer.eos_token, tokenizer.decode(encoded))
    asserts.assert_equal(in_vocab, tokenizer.decode(encoded, remove_tokens = True))

def test_decode_ids(asserts, tokenizer):
    # a single id -> its symbol ; a list -> the char-level (sep = '') concatenation
    asserts.assert_equal('H', tokenizer.decode_ids(tokenizer['H']))
    asserts.assert_equal('Hello', tokenizer.decode_ids([tokenizer[c] for c in 'Hello']))

def test_decode_skip_padding(asserts, tokenizer):
    h, i = tokenizer['H'], tokenizer['i']
    pad  = tokenizer.blank_token_idx

    # an isolated padding token is skipped, decoding continues
    asserts.assert_equal('Hi', tokenizer.decode([h, pad, i]))
    # two consecutive padding tokens mark the end -> the rest is dropped
    asserts.assert_equal('H', tokenizer.decode([h, pad, pad, i]))

def test_decode_keep_padding(asserts, tokenizer):
    h, i = tokenizer['H'], tokenizer['i']
    pad  = tokenizer.blank_token_idx
    asserts.assert_equal(
        'H' + tokenizer.blank_token + 'i', tokenizer.decode([h, pad, i], skip_padding = False)
    )

def test_decode_remove_tokens(asserts, tokenizer):
    """ Regression : ``remove_tokens`` must strip special tokens *and* padding. """
    encoded = tokenizer.encode('Hi', cleaned = True, return_type = 'list')   # [sos, H, i, eos]
    asserts.assert_equal('Hi', tokenizer.decode(encoded, remove_tokens = True))
    asserts.assert_equal(
        'Hi', tokenizer.decode(encoded + [tokenizer.blank_token_idx], remove_tokens = True)
    )

def test_decode_batched(asserts, tokenizer):
    expected = ['<s>Hi</s>', '<s>Hello</s>']
    # padded 2-D `np.ndarray` -> list of strings (the trailing padding is trimmed)
    asserts.assert_equal(
        expected, tokenizer.decode(tokenizer.encode(['Hi', 'Hello'], cleaned = True, return_type = 'np'))
    )
    # list of (ragged) id lists -> list of strings
    asserts.assert_equal(
        expected, tokenizer.decode(tokenizer.encode(['Hi', 'Hello'], cleaned = True, return_type = 'list'))
    )

def test_decode_argmax_logits(asserts, tokenizer):
    """ A float ``(T, vocab)`` array is ``argmax``-ed before decoding ; batched -> list. """
    ids    = tokenizer.encode('Hello', cleaned = True, add_sos_and_eos = False, return_type = 'list')
    logits = np.full((len(ids), len(tokenizer)), -1., dtype = np.float32)
    for t, idx in enumerate(ids):
        logits[t, idx] = 1.

    asserts.assert_equal('Hello', tokenizer.decode(logits))
    asserts.assert_equal(['Hello', 'Hello'], tokenizer.decode(np.stack([logits, logits])))


# --- serialization round-trip ------------------------------------------------------

def test_config_roundtrip(asserts, tokenizer):
    clone = Tokenizer(** tokenizer.get_config())
    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), clone.encode(text))

def test_save_load(asserts, tokenizer, temp_dir):
    path = os.path.join(temp_dir, 'test_tokenizer.json')
    tokenizer.save(path)
    loaded = Tokenizer.load_from_file(path)

    for text in _default_texts:
        asserts.assert_equal(tokenizer.encode(text), loaded.encode(text))

    encoded = tokenizer.encode('Hello', cleaned = True)
    asserts.assert_equal(tokenizer.decode(encoded), loaded.decode(encoded))


# --- tensorflow graph (`tf.data` pipeline) -----------------------------------------

@pytest.mark.tensorflow
def test_tf_function(asserts, tokenizer):
    import tensorflow as tf

    pipe = tf.data.Dataset.from_tensor_slices(_default_texts).map(tokenizer.encode)
    for text, encoded in zip(_default_texts, pipe):
        asserts.assert_tf_tensor(encoded)
        asserts.assert_equal(tokenizer.encode(text), encoded)

    asserts.assert_equal(
        tokenizer.encode(_default_texts, return_type = 'np'),
        tf.function(tokenizer.encode)(tf.reshape(_default_texts, [-1]), shape = (None, None)),
    )


# --- `transformers` round-trip (heavy / network) -----------------------------------

@pytest.mark.slow
@pytest.mark.parametrize('model_name,use_sos_and_eos,add_eos', [
    ('facebook/bart-large',          None,  None),
    ('moussaKam/barthez',            None,  None),
    ('bert-base-uncased',            None,  None),
    ('bert-base-cased',              None,  None),
    ('gpt2',                         False, None),
    ('tiiuae/falcon-7b',             False, None),
    ('google/flan-t5-large',         None,  None),
    ('BAAI/bge-m3',                  None,  None),
    ('bofenghuang/vigostral-7b-chat', True, False),
])
def test_transformers_tokenizer(asserts, model_name, use_sos_and_eos, add_eos):
    transformers = pytest.importorskip('transformers')

    reference = transformers.AutoTokenizer.from_pretrained(model_name)
    encoder   = Tokenizer.from_transformers_pretrained(model_name)

    if use_sos_and_eos is not None:
        encoder.use_sos_and_eos = use_sos_and_eos

    for sent in _default_texts:
        asserts.assert_equal(reference.tokenize(sent), encoder.tokenize(sent))
        asserts.assert_equal(
            reference(sent)['input_ids'], encoder.encode(sent, add_eos = add_eos)
        )
