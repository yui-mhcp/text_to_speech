# :yum: Text To Speech (TTS)

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![License](https://img.shields.io/badge/licence-AGPL--3.0-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)

This repository provides a modular Text-To-Speech (TTS) toolkit built on the yui-mhcp `BaseModel` interface, featuring the `Tacotron-2` synthesizer, the `WaveGlow` vocoder, and the `SV2TTS` extension for zero-shot voice cloning.

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers                   : custom layer implementations (attention, embeddings, invertible conv, ...)
│   ├── transformers             : transformer architecture implementations (bart, bert, gpt2, mistral, t5, whisper, ...)
│   ├── current_blocks.py        : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── generation_utils.py      : utilities for text and sequence generation
│   ├── hparams.py               : hyperparameter management
│   ├── simple_models.py         : classical models such as CNN / RNN / MLP and siamese
│   ├── tacotron2_arch.py        : Tacotron-2 synthesizer architecture (+ multi-speaker variant)
│   └── waveglow_arch.py         : WaveGlow vocoder architecture
├── custom_train_objects    : custom objects used in training / testing
│   ├── callbacks                : callbacks loading and implementations
│   ├── generators               : custom data generators
│   │   ├── file_cache_generator.py : cache-based generator to speed up audio processing
│   │   └── ge2e_generator.py       : GE2E generator for speaker-encoder training
│   ├── losses                   : loss functions
│   │   ├── ctc_loss.py             : CTC loss
│   │   ├── loss_with_multiple_outputs.py : utility class allowing losses to output additional metrics
│   │   ├── tacotron_loss.py        : custom Tacotron-2 loss
│   │   └── text_loss.py            : text (token-level) loss
│   ├── metrics                  : metrics loading and implementations
│   ├── optimizers               : optimizer loading + custom learning-rate schedulers
│   ├── checkpoint_manager.py    : handle model checkpoint management
│   └── history.py               : main History class to track training statistics / config
├── loggers                 : custom utilities for the `logging` module*
├── models                  : main directory for model classes
│   ├── core                     : the `BaseModel` foundation (mixins + saving / migration utils)*
│   ├── tts                      : Text-To-Speech models
│   │   ├── sv2tts_tacotron2.py     : SV2TTS multi-speaker synthesizer
│   │   ├── tacotron2.py            : Tacotron-2 synthesizer
│   │   └── waveglow.py             : WaveGlow vocoder (both pytorch and tensorflow)
│   └── weights_converter.py     : utilities to convert weights between different models
├── pretrained_models       : pretrained checkpoints (unzip the downloaded models here)
├── tests                   : `pytest` suite mirroring the `utils` / `loggers` / model tree*
├── utils                   : shared data-processing / keras utilities*
├── example_outputs             : some pre-computed audios (cf the `text_to_speech` notebook)
├── text_to_speech.ipynb        : notebook demonstrating model creation + TTS features
├── CHANGELOG.md                : project-specific changelog
├── CITATION.cff                : citation metadata
├── CONTRIBUTING.md             : contribution guidelines
├── INSTALLATION.md             : shared GPU environment installation guide
├── LICENCE                     : project license file
├── pyproject.toml              : project metadata, dependencies and tooling config
├── __version__.py              : single-sourced version string
└── README.md                   : this file
```

\* The `loggers`, `utils`, `models/core` and most of the `tests` tree are the shared foundation maintained in [the data_processing](https://github.com/yui-mhcp/data_processing) and [base_dl_project](https://github.com/yui-mhcp/base_dl_project) repositories — check them for more information about the unextended modules / structure / main classes.

\* Check the [encoders](https://github.com/yui-mhcp/encoders) project for more information about the speaker encoder used by the `SV2TTS` models.

## Available features

- **Text-To-Speech** (module `models.tts`) :

| Feature   | Function / class   | Description |
| :-------- | :---------------- | :---------- |
| Text-To-Speech    | `tts`             | perform TTS on text you want with the model you want  |
| stream            | `stream`          | perform TTS on text you enter |

The `text_to_speech` notebook provides a concrete demonstration of the `tts` function.

## Available models

### Model architectures

Available architectures: 
- `Synthesizer`:
    - [Tacotron2](https://arxiv.org/abs/1712.05884) with extensions for multi-speaker (by ID or `SV2TTS`)
    - [SV2TTS](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf) extension of the Tacotron2 architecture for multi-speaker based on speaker's embeddings\*
- `Vocoder`:
    - [Waveglow](https://arxiv.org/abs/1811.00002)

The SV2TTS models are fine-tuned from pretrained Tacotron2 models, by using the *partial transfer learning* procedure (see below for details), which speeds up the training significantly.

### Model weights

| Name      | Language  | Dataset   | Synthesizer   | Vocoder   | Speaker Encoder   | Trainer   | Weights   |
| :-------: | :-------: | :-------: | :-----------: | :-------: | :---------------: | :-------: | :-------: |
| pretrained_tacotron2  | `en`      | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)   | `Tacotron2`   | `WaveGlow`    | / | [NVIDIA](https://github.com/NVIDIA)   | [Google Drive](https://drive.google.com/file/d/1mnhPgOE8IrQ3cTtfwOScZEn3aFZvaZG7/view?usp=sharing)  |
| tacotron2_siwis   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full)   | `Tacotron2`   | `WaveGlow`    | / | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1r3Dzu5V1A34-StUeKPt0Fl_RoQohu8t_/view?usp=sharing)  |
| sv2tts_tacotron2_256  | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1-WWfmQs7pGRQpcZPI6mn9c4FTWnrHZem/view?usp=sharing) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1at9bYsAoazqMccDBXW089DjivMS1Nb2x/view?usp=sharing)  |
| sv2tts_siwis  | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1-WWfmQs7pGRQpcZPI6mn9c4FTWnrHZem/view?usp=sharing) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1GESyvKozvWEj7nfC7Qin2xuMJrL4pqTS/view?usp=sharing)  |
| sv2tts_tacotron2_256_v2   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1UK44V7C-hlj_pziAuQnJnHxAwLyoxcjt/view?usp=share_link)  |
| sv2tts_siwis_v2   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1BaCSuWeydNj5z0b6dgKddPUDM2j4rPvu/view?usp=share_link)  |

Models must be unzipped in the `pretrained_models/` directory!

**Important Note**: These links will be updated in a future version, and the converted keras weights of `WaveGlow` will also be added. 

## Installation and usage

See [the installation guide](INSTALLATION.md) for a step-by-step setup of the shared GPU environment (NVIDIA driver, CUDA, `mamba` and the deep-learning backends) :smile:

Here is a summary of the installation procedure, once your python environment is ready :
1. Clone this repository : `git clone https://github.com/yui-mhcp/text_to_speech.git`
2. Go to the root of this repository : `cd text_to_speech`
3. Install the package with a backend : `pip install -e .[tf]` (or `pip install -e .[torch]`)
4. Open the `text_to_speech` notebook and follow the instructions !

Unlike the pure-`numpy` `data_processing` socle, this project runs real models, so it needs a **keras backend**. The **audio** and **text** stacks are core to TTS and are already bundled in the base install — you only pick the backend matching your setup :

```bash
pip install -e .[tf]         # keras + tensorflow
pip install -e .[torch]      # keras + torch
pip install -e .[keras]      # alias of [tf]

# optional helpers :
pip install -e .[datasets]   # utils/datasets (tensorflow-datasets, pandas) — to (re)train
pip install -e .[dev]        # test tooling (pytest & plugins)

# extras can be combined :
pip install -e .[tf,datasets]
```

The backend is selected at runtime through the `KERAS_BACKEND` environment variable (e.g., `os.environ['KERAS_BACKEND'] = 'tensorflow'`), as shown at the top of the example notebook. You may also have to install `ffmpeg` for audio loading/saving.

### Quickstart

```python
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from models.tts import tts

# Perform TTS with the default English model (Tacotron-2 + WaveGlow) ...
text = "Hello world ! I hope you will enjoy this funny API for Text-To-Speech !"
_ = tts(text, lang = 'en', directory = 'example_outputs/en', display = True)

# ... or with the default French model (SV2TTS voice cloning)
_ = tts("Bonjour tout le monde !", lang = 'fr', directory = 'example_outputs/fr', display = True)
```

See the `text_to_speech.ipynb` notebook for the full walkthrough (model creation, single / multi-speaker synthesis, long-text and streaming inference). :smile:

### Testing

The tests use [`pytest`](https://docs.pytest.org/) and live in the `tests/` directory, mirroring the `utils` / `loggers` and model tree.

```bash
pip install -e .[dev]        # pytest, pytest-cov, pytest-xdist, pytest-timeout

pytest                       # run the whole suite
pytest -n auto               # run in parallel (pytest-xdist)
pytest -m "not slow"         # skip the slow / heavy tests
pytest --cov                 # run with coverage report
pytest tests/models          # run a single subpackage
```

Tests are annotated with markers (declared in `pyproject.toml`) so the suite adapts to your environment : the `tensorflow`, `torch`, `keras`, `cv2` and `gpu` markers are **auto-skipped** when the corresponding dependency (or hardware) is missing, meaning you can run the tests for the backend you installed without pulling every dependency.

## TO-DO list:

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Add `batch_size` support for `vocoder inference`
- [x] Add pretrained `SV2TTS` weights
- [x] Add document parsing to perform `TTS` on document (in progress)
- [x] Train a `SV2TTS` model based on an encoder trained with the `GE2E` loss
- [x] Add support for long text inference
- [x] Add support for streaming inference
- [ ] Update the pretrained models links + add the `WaveGlow` model
- [ ] Update the speaker encoder module
- [ ] Update the `Google Colab` demo
- [ ] Update the training notebooks
- [ ] Train new `SV2TTS` models based on the optimized pre-processing codes
- [ ] Add dedicated document-based `TTS`


## Multi-speaker Text-To-Speech

There are multiple ways to enable `multi-speaker` speech synthesis:
1. Use a `speaker ID` that is embedded by a learnable `Embedding` layer. The speaker embedding is then learned during training. 
2. Use a `Speaker Encoder (SE)` to embed audio from the reference speaker. This is often referred to as `zero-shot voice cloning`, as it only requires a sample from the speaker (without training). 
3. Recently, a new `prompt-based` strategy has been proposed to control the speech with prompts. 

### Automatic voice cloning with the `SV2TTS` architecture

Note: In the next paragraphs, `encoder` refers to the `Tacotron Encoder` part (that encodes the input text), while `SE` refers to a `speaker encoder` model (detailed below).

#### The basic intuition

The `Speaker Encoder-based Text-To-Speech` is inspired from the "[From Speaker Verification To Text-To-Speech (SV2TTS)](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf)" paper. The authors have proposed an extension of the `Tacotron-2` architecture to include information about the speaker's voice. 

Here is a short overview of the proposed procedure:
1. Train a model to identify speakers based on short audio samples: the `speaker verification` model. This model takes as input an audio sample (5-10 sec) from a speaker and encodes it into a *d*-dimensional vector, named the `embedding`. This embedding aims to capture relevant information about the speaker's voice (e.g., `frequencies`, `rhythm`, `pitch`, etc.). 
2. This pre-trained `Speaker Encoder (SE)` is then used to encode the voice of the speaker to clone.
3. The produced embedding is then concatenated with the output of the `Tacotron-2` encoder part, such that the `Decoder` has access to both the encoded text and the speaker embedding.

The objective is that the `Decoder` will learn to use the `speaker embedding` to copy its prosody/intonation/etc. to read the text with the voice of this speaker.

#### Limitations and solutions

There are some limitations with the above approach: 
- Perfect generalization to new speakers is very difficult, as it would require large datasets with many speakers. 
- The audio should not have any noise/artifacts to avoid noisy synthetic audios.
- The `Speaker Encoder` has to correctly separate speakers and encode their voice in a meaningful way for the synthesizer.

To tackle these limitations, the proposed solution is to perform a 2-step training:
- First train a *low-quality* multi-speakers model on the `CommonVoice` database. This is one of the largest multilingual databases for audio, at the cost of noisy/variable quality audios. This is therefore not suitable to train good quality models, whereas pre-processing still helps to obtain intelligible audios. 
- Once a multi-speaker model is trained, a single-speaker database with a limited amount of good quality data can be used to fine-tune the model on a single speaker. This allows the model to learn faster, with only a limited amount of good quality data, and to produce really good quality audios!

#### The Speaker Encoder (SE)

The SE part should be able to differentiate speakers and embed (encode in a 1-D vector) them in a *meaningful* way. 

The model used in the paper is a 3-layer `LSTM` model with a normalization layer trained with the [GE2E](https://ieeexplore.ieee.org/abstract/document/8462665) loss. The major limitation is that training this model is **really slow** and took 2 weeks on 4 GPUs in CorentinJ's master thesis (cf. his [GitHub](https://github.com/CorentinJ/Real-Time-Voice-Cloning)).

This project proposes a simpler architecture based on `Convolutional Neural Networks (CNN)`, which is much faster to train compared to `LSTM` networks. Furthermore, the `Euclidean` distance has been used rather than the `cosine` metric, which has shown faster convergence. Additionally, a custom cache-based generator is proposed to speed up audio processing. These modifications allowed training a 99% accuracy model within 2-3 hours on a single `RTX 3090` GPU!


## The *partial* Transfer Learning procedure

In order to avoid training a SV2TTS model from scratch, which would be completely impossible on a single GPU, a new `partial transfer learning` procedure is proposed. 

This procedure takes a pre-trained model with a slightly different architecture and transfers all the common weights (like in regular transfer learning). For the layers with different weight shapes, only the common part is transferred, while the remaining weights are initialized to zeros. This results in a new model with different weights that mimics the behavior of the original model.

In the `SV2TTS` architecture, the speaker embedding is passed to the recurrent layer of the `Tacotron2 decoder`. This results in a different input shape, making the layer weights matrix different. The partial transfer learning allows us to initialize the model such that it replicates the behavior of the original single-speaker `Tacotron2` model!

## Notes and references 

### GitHub projects

The code for this project is a mixture of multiple GitHub projects, to have a fully modular `Tacotron-2` implementation:
- [NVIDIA's repository (tacotron2 / waveglow)](https://github.com/NVIDIA): The base pretrained models are inspired from this repository. 
- [The TFTTS project](https://github.com/TensorSpeech/TensorflowTTS): Some inference optimizations are inspired from their `dynamic decoder` implementation, which has now been optimized and updated to be `Keras 3` compatible.
- [CorentinJ's Real-Time Voice cloning project](https://github.com/CorentinJ/Real-Time-Voice-Cloning): The provided `SV2TTS` architecture is inspired from this repository, with small differences and optimizations. 

### Papers

- [Tacotron 2](https://arxiv.org/abs/1712.05884): The original Tacotron2 paper
- [Waveglow](https://arxiv.org/abs/1811.00002): The original WaveGlow paper
- [Transfer learning from Speaker Verification to Text-To-Speech](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf): Original paper for SV2TTS variant
- [Generalized End-to-End loss for Speaker Verification](https://ieeexplore.ieee.org/abstract/document/8462665): The GE2E Loss paper (used for speaker encoder in the SV2TTS architecture)


## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENCE](LICENCE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp,
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```
