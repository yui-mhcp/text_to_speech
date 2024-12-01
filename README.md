# :yum: Text To Speech (TTS)

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

## Project structure

```bash

├── custom_architectures
│   ├── tacotron2_arch.py       : Tacotron-2 synthesizer architecture
│   └── waveglow_arch.py        : WaveGlow vocoder architecture
├── custom_layers
├── custom_train_objects
│   ├── losses
│   │   └── tacotron_loss.py    : custom Tacotron2 loss
├── example_outputs         : some pre-computed audios (cf the `text_to_speech` notebook)
├── loggers
├── models
│   ├── encoder             : the `AudioEncoder` is used as speaker encoder for the SV2TTS model*
│   ├── tts
│   │   ├── sv2tts_tacotron2.py : SV2TTS main class
│   │   ├── tacotron2.py        : Tacotron2 main class
│   │   ├── vocoder.py          : main functions for complete inference
│   │   └── waveglow.py         : WaveGlow main class (both pytorch and tensorflow)
├── pretrained_models
├── unitests
├── utils
├── example_fine_tuning.ipynb
├── example_sv2tts.ipynb
├── example_tacotron2.ipynb
├── example_waveglow.ipynb
└── text_to_speech.ipynb
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

\* Check the [encoders](https://github.com/yui-mhcp/encoders) project for more information about the `models/encoder` module

## Available features

- **Text-To-Speech** (module `models.tts`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| Text-To-Speech    | `tts`             | perform TTS on text you want with the model you want  |
| stream            | `tts_stream`      | perform TTS on text you enter |
| TTS logger        | `loggers.TTSLogger`   | converts `logging` logs to voice and play it |

The `text_to_speech` notebook provides a concrete demonstration of the `tts` function

## Available models

### Model architectures

Available architectures : 
- `Synthesizer` :
    - [Tacotron2](https://arxiv.org/abs/1712.05884) with extensions for multi-speaker (by ID or `SV2TTS`)
    - [SV2TTS](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf) extension of the Tacotron2 architecture for multi-speaker based on speaker's embeddings\*
- `Vocoder` :
    - [Waveglow](https://arxiv.org/abs/1811.00002)

The SV2TTS models are fine-tuned from pretrained Tacotron2 models, by using the *partial transfer learning* procedure (see below for details), which speeds up a lot the training.

### Model weights

| Name      | Language  | Dataset   | Synthesizer   | Vocoder   | Speaker Encoder   | Trainer   | Weights   |
| :-------: | :-------: | :-------: | :-----------: | :-------: | :---------------: | :-------: | :-------: |
| pretrained_tacotron2  | `en`      | [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)   | `Tacotron2`   | `WaveGlow`    | / | [NVIDIA](https://github.com/NVIDIA)   | [Google Drive](https://drive.google.com/file/d/1mnhPgOE8IrQ3cTtfwOScZEn3aFZvaZG7/view?usp=sharing)  |
| tacotron2_siwis   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full)   | `Tacotron2`   | `WaveGlow`    | / | [me](https://github.com/yui-mhcp) | [Google Drive](https://drive.google.com/file/d/1r3Dzu5V1A34-StUeKPt0Fl_RoQohu8t_/view?usp=sharing)  |
| sv2tts_tacotron2_256  | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1-WWfmQs7pGRQpcZPI6mn9c4FTWnrHZem/view?usp=sharing) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1at9bYsAoazqMccDBXW089DjivMS1Nb2x/view?usp=sharing)  |
| sv2tts_siwis  | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1-WWfmQs7pGRQpcZPI6mn9c4FTWnrHZem/view?usp=sharing) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1GESyvKozvWEj7nfC7Qin2xuMJrL4pqTS/view?usp=sharing)  |
| sv2tts_tacotron2_256_v2   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full), [VoxForge](http://www.voxforge.org/), [CommonVoice](https://commonvoice.mozilla.org/fr/datasets)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1UK44V7C-hlj_pziAuQnJnHxAwLyoxcjt/view?usp=share_link)  |
| sv2tts_siwis_v2   | `fr`      | [SIWIS](https://datashare.ed.ac.uk/handle/10283/2353?show=full)   | `SV2TTSTacotron2`   | `WaveGlow`    | [Google Drive](https://drive.google.com/file/d/1bzj9412l0Zje3zLaaqGOBNaQRBYLVO2q/view?usp=share_link) | [me](https://github.com/yui-mhcp)  | [Google Drive](https://drive.google.com/file/d/1BaCSuWeydNj5z0b6dgKddPUDM2j4rPvu/view?usp=share_link)  |

Models must be unzipped in the `pretrained_models/` directory !

**Important Note** : the `NVIDIA` models available on `torch hub` requires a compatible GPU with the correct configuration for `pytorch`. It is the reason why the both models are provided in the expected `keras` checkpoint :smile:

The `sv2tts_siwis` models are fine-tuned version of `sv2tts_tacotron2_256` on the `SIWIS` (single-speaker) dataset. Fine-tuning a multi-speaker on a single-speaker dataset tends to improve the stability, and to produce a voice with more intonation, compared to simply training the single-speaker model. 

## Usage and demonstration

### Demonstration

A [Google Colab](https://colab.research.google.com) demo is available [at this link](https://colab.research.google.com/drive/18odkKiBl8sm4Ba9S3TM1V128uO9HbDn0?usp=sharing) !

You can also find some audio generated in `example_outputs/`, or directly in the Colab notebook ;)

### Installation and usage

Check [this installagion guide](https://github.com/yui-mhcp/yui-mhcp/blob/main/INSTALLATION.md) for the step-by-step instructions !

You may have to install `ffmpeg` for audio loading / saving.

## TO-DO list :

- [x] Make the TO-DO list
- [x] Comment the code
- [x] Add pretrained weights for French
- [x] Make a `Google Colab` demonstration
- [x] Implement WaveGlow in `tensorflow 2.x`
- [x] Add `batch_size` support for `vocoder inference`
- [x] Add pretrained `SV2TTS` weights
- [x] Add a `similarity loss` to test a new training procedure for single-speaker fine-tuning
- [ ] Add document parsing to perform `TTS` on document (in progress)
- [ ] Add new languages support
- [ ] Add new TTS architectures / models
- [x] Train a `SV2TTS` model based on an encoder trained with the `GE2E` loss
- [x] **Experimental** add support for long text inference
- [ ] Add support for streaming inference


## Multi-speaker Text-To-Speech

There are multiple ways to enable `multi-speaker` speech synthesis :
1. Use a `speaker ID` that is embedded by a learnable `Embedding` layer. The speaker embedding is then learned during training. 
2. Use a `Speaker Encoder (SE)` to embed audio from the reference speaker. This is often referred as `zero-shot voice cloning`, as it only requires a sample from the speaker (without training). 
3. Recently, a new `prompt-based` strategy has been proposed to control the speech with prompts. 

### Automatic voice cloning with the `SV2TTS` architecture

Note : in the next paragraphs, `encoder` refers to the `Tacotron Encoder` part, while `SE` refers to a `speaker encoder` model (detailed below).

#### The basic intuition

The `Speaker Encoder-based Text-To-Speech` is inspired from the "[From Speaker Verification To Text-To-Speech (SV2TTS)](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf)" paper. The authors have proposed an extension of the `Tacotron-2` architecture to include information about the speaker's voice. 

Here is a short overview of the proposed procedure :
1. Train a model to identify speakers based on short audio samples : the `speaker verification` model. This model basically takes as input an audio sample (5-10 sec) from a speaker, and encodes it on a *d*-dimensional vector, named the `embedding`. This embedding aims to capture relevant information about the speaker's voice (e.g., `frequencies`, `rythm`, `pitch`, ...). 
2. This pre-trained `Speaker Encoder (SE)` is then used to encode the voice of the speaker to clone.
3. The produced embedding is then concatenated with the output of the `Tacotron-2` encoder part, such that the `Decoder` has access to both the encoded text and the speaker embedding.

The objective is that the `Decoder` will learn to use the `speaker embedding` to copy its prosody / intonation / ... to read the text with the voice of this speaker.

#### Limitations and solutions

There are some limitations with the above approach : 
- A perfect generalization to new speakers is really difficult, as it would require large datasets with many speakers. 
- The audio should not have any noise / artifacts to avoid noisy synthetic audios.
- The `Speaker Encoder` has to correctly separate speakers, and encode their voice in a meaningful way for the synthesizer.

To tackle these limitations, the proposed solution is to perform a 2-step training :
- First train a *low-quality* multi-speakers model on the `CommonVoice` database. This is one of the largest multilingual database for audio, at the cost of noisy / variable quality audios. This is therefore not suitable to train good quality models, whereas pre-processing still helps to obtain intelligible audios. 
- Once a multi-speaker model is trained, a single-speaker database with few good quality data can be used to fine-tune the model on a single speaker. This allows the model to learn faster, with only limited amount of good quality data, and to produce really good quality audios !

#### The Speaker Encoder (SE)

The SE part should be able to differentiate speakers, and embed (encode in a 1-D vector) them in a *meaningful* way. 

The model used in the paper is a 3-layer `LSTM` model with a normalization layer trained with the [GE2E](https://ieeexplore.ieee.org/abstract/document/8462665) loss. The major limitation is that training this model is **really slow**, and took 2 weeks on 4 GPU's in the CorentinJ master thesis (cf his [github](https://github.com/CorentinJ/Real-Time-Voice-Cloning))

This project proposes a simpler architecture based on `Convolutional Neural Networks (CNN)`, which is much faster to train compared to `LSTM` networks. Furthermore, the `euclidian` distance has been used rather than the `cosine` metric, which has shown faster convergence. Additionally, a custom cache-based generator is proposed to speed up audio processing. These modifications allowed to train a 99% accuracy model within 2-3 hours on a single `RTX 3090` GPU !


## The *partial* Transfer Learning procedure

In order to avoid training a SV2TTS model from scratch, which would be completely impossible on a single GPU, a new `partial transfer learning` procedure is proposed. 

This procedure takes a pre-trained model with a slightly different architecture, and transfer all the common weights (like in regular transfer learning). For the layers with different weights shape, only the common part is transfered, while the remaining weights are initialized to zeros. This result in a new model with different weights to mimic the behavior of the original model.

In the `SV2TTS` architecture, the speaker embedding is passed to the recurrent layer of the `Tacotron2 decoder`. This results in a different input shape, making the layer weights matrix different. The partial transfer learning allows to nitialize the model such that it replicates the behavior of the original single-speaker `Tacotron2` model !

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

The code for this project is a mixture of multiple GitHub projects, to have a fully modulable `Tacotron-2` implementation
- [NVIDIA's repository (tacotron2 / waveglow)](https://github.com/NVIDIA) : the base pretrained model is are inspired from this repository. 
- [The TFTTS project](https://github.com/TensorSpeech/TensorflowTTS) : some inference optimizations are inspired from their `dynamic decoder` implementation, which has now been optimized and updated to be `Keras 3` compatible.
- [CorentinJ's Real-Time Voice cloning project](https://github.com/CorentinJ/Real-Time-Voice-Cloning) : the provided `SV2TTS` architecture is inspired from this repository, with small differences and optimizations. 

Papers : 
- [Tacotron 2](https://arxiv.org/abs/1712.05884) : the original Tacotron2 paper
- [Waveglow](https://arxiv.org/abs/1811.00002) : the original WaveGlow paper
- [Transfer learning from Speaker Verification to Text-To-Speech](https://papers.nips.cc/paper/2018/file/6832a7b24bc06775d02b7406880b93fc-Paper.pdf)) : original paper for SV2TTS variant
- [Generalized End-to-End loss for Speaker Verification](https://ieeexplore.ieee.org/abstract/document/8462665) : the GE2E Loss paper (used for speaker encoder in the SV2TTS architecture)
