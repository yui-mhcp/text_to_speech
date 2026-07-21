# :yum: CHANGELOG :yum:

This file tracks the major updates of the [text_to_speech](https://github.com/yui-mhcp/text_to_speech) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the Text-To-Speech stack, now built on top of the refactored `BaseModel` foundation (see the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) and the shared `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .` and pick a keras backend extra (`tf` or `torch`). Unlike the socle projects, the **audio** and **text** stacks are core to TTS and ship in the base install (no extra needed)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The minimum supported Python is now `3.12` (backend wheels), and the version is single-sourced in `__version__`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured : the old `models/interfaces/base_*_model.py` classes are gone. `BaseModel` now lives in `models/core/base_model.py` and is composed from small, reusable **mixins** in `models/core/mixins/` (`audio`, `image`, `text`, `classification`, `processing`, `checkpoint`, `training`)
- **[BREAKING CHANGE]** model-core internals (saving / loading, legacy migration, description) have moved to `models/core/utils/{saving, migration, describe}.py`
- **[BREAKING CHANGE]** the stale `models/encoder` reference has been removed ; the speaker encoder used by `SV2TTS` lives in the dedicated [encoders](https://github.com/yui-mhcp/encoders) project

### Minor updates

- `models/tts` keeps the three synthesizer / vocoder classes : `Tacotron2`, `SV2TTSTacotron2` and `WaveGlow` (both pytorch and tensorflow weights), exposed through the `tts` and `stream` entry-points
- The dynamic decoder used at inference has been optimized and updated to be `Keras 3` compatible
- `architectures` keeps the TTS-specific `tacotron2_arch` / `waveglow_arch`, the custom `layers` (`location_sensitive_attention`, `concat_embedding`, `invertible_conv`, `custom_rnn_dropout_cell`, ...) and the shared transformer architectures (`bart`, `bert`, `gpt2`, `mistral`, `t5`, `whisper`, ...)
- `custom_train_objects` cleanup : TTS / speaker-encoder losses (`tacotron_loss`, `ctc_loss`, `text_loss`), generators (`file_cache_generator`, `ge2e_generator`), callbacks, metrics and optimizers
- The `text_to_speech.ipynb` notebook has been reworked around the new `models.tts.tts` / `stream` interface
