# Research code availability

SurpriseNet is preserved as a 2021 research snapshot. The maintained part of this repository is the static, precomputed [web sample explorer](docs/). The Python files expose useful implementation details, but the repository does not contain a verified end-to-end path from raw data to training or from a new melody to generated harmony.

## What is available

- CVAE and SurpriseNet-related model definitions under [`model/`](model/).
- Historical data, decoding, evaluation, and tonal utilities under [`utils/`](utils/) and [`tonal.py`](tonal.py).
- The archival Python 3.6 Conda specification in [`environment.yml`](environment.yml).
- A dependency-free static explorer of 39 precomputed audio-visual samples under [`docs/`](docs/).

## What is not included

- A reproducible raw-data preprocessing command and versioned output schema.
- A supported SurpriseNet training command.
- A compatible pretrained checkpoint.
- A supported inference command for arbitrary MIDI or melody input.
- A modern, validated Python or Apple Silicon/MPS environment.

The static explorer does not call the Python model and has no inference backend. It selects among checked-in MP4 samples, so hosting it on GitHub Pages does not add custom-melody generation.

## Filename and environment history

The setup questions in the original README referred to files that later moved or disappeared:

- Commit [`2413aa9`](https://github.com/scmvp301135/SurpriseNet/commit/2413aa9) renamed `surprisenet_train.py` to `train.py` and `surprisenet_inference.py` to `test.py` without updating the old README instructions. The inference file was later removed; the current `eval.py` came from a different historical CVAE evaluation script.
- Commit [`6b5ff32`](https://github.com/scmvp301135/SurpriseNet/commit/6b5ff32) removed the old `requirements.txt` when `environment.yml` was added. Recreating that obsolete requirements file would not make the current snapshot reproducible.
- The old `with -epoch` example was not valid syntax for the checked-in command-line parser.

These facts explain the missing filenames; they do not establish that the renamed files are runnable replacements.

## Verified blockers

The current source has incompatible producer and consumer contracts:

- [`utils/dataloader.py`](utils/dataloader.py) does not create every array later read by its dataset object, does not populate surprise contours, and contains incomplete conversion paths.
- [`train.py`](train.py) references trainer attributes and validation arrays that are never initialized, combines incompatible data-loader options, and does not match the checked-in YAML structure.
- [`eval.py`](eval.py) imports removed modules, reads fixed prepared arrays rather than a user-supplied melody, and requires artifacts that are absent from the repository.
- No Git commit, tag, LFS object, or repository release contains a compatible pretrained SurpriseNet checkpoint.

Installing the archival dependencies does not resolve those code and artifact gaps. The historical preprocessing approach also materializes very large intermediate arrays that can exceed available memory. Failure on a modern Mac is therefore possible for both correctness and memory reasons, not merely because of Apple MPS support.

## Restoration scope

A future reproducibility release should be treated as a separate engineering project. At minimum it would need:

1. A deterministic preprocessing CLI with a documented, versioned schema and small redistributable fixture.
2. A validated dependency lock and CI smoke test on supported hardware.
3. A corrected training entry point with a one-batch and one-epoch integration test.
4. A published checkpoint with checksum, license, model card, and matching vocabulary metadata.
5. A user-input adapter that specifies MIDI quantization, tempo, key, melody extraction, and output behavior.
6. An end-to-end test that produces a harmony from a tiny input without relying on unpublished files.

Until that work exists, `train.py`, `eval.py`, and `environment.yml` should be read as incomplete historical artifacts rather than supported setup instructions.
