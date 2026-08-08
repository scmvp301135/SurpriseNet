# SurpriseNet

SurpriseNet is a user-controlled conditional variational autoencoder for melody harmonization. It conditions generated chord progressions on a surprise contour selected by the listener.

[Web sample explorer](https://scmvp301135.github.io/SurpriseNet/) · [Paper](https://arxiv.org/abs/2108.00378) · [ISMIR 2021](https://ismir2021.ismir.net/)

## Web sample explorer

The sample explorer is a static, precomputed listening demo under [`docs/`](docs/). It does not run SurpriseNet and does not require a backend. GitHub Pages can serve it directly. A hosted inference service such as Hugging Face Spaces would only be needed for user-submitted melodies or real-time generation.

The explorer compares:

- 3 melody excerpts
- 6 user-controlled surprise contours
- Ground truth, SurpriseNet, and Weighted SurpriseNet outputs
- 39 audio-visual samples in total

Only the selected video is downloaded after the listener presses **Load and play**. The initial page load does not fetch the 51.77 MiB sample collection.

Run the site locally with Python:

```bash
python3 -m http.server 8000 --directory docs
```

Then open <http://localhost:8000>.

Run the dependency-free web and documentation checks with Node.js:

```bash
node --test tests/repository-docs.test.mjs tests/web-demo.test.mjs
```

The Pages workflow publishes `docs/` after changes reach `master`. In the repository settings, select **GitHub Actions** as the Pages source once. A separate website branch is no longer required after this refactor is merged and the publishing source is switched.

## Research code status

The model, data-loader, and evaluation files are retained as an incomplete historical artifact. They are useful for reading the 2021 implementation, but they do not form a supported end-to-end training or inference release.

| Capability | Status |
| --- | --- |
| Static precomputed listening demo | Available |
| Historical model definitions | Available |
| Reproducible preprocessing | Not available |
| Supported SurpriseNet training CLI | Not included |
| Pretrained checkpoint | Not included |
| Arbitrary-melody inference | Not included |

The old `surprisenet_train.py` and `surprisenet_inference.py` filenames were renamed during an unfinished refactor. The current `train.py` and `eval.py` are incomplete historical artifacts, not replacement commands that are known to run. The web explorer only plays precomputed samples; it cannot harmonize a new melody.

The historical Conda specification is in [`environment.yml`](environment.yml). It targets Python 3.6 and requires separately prepared data that is not reproducible from the checked-in files. Treat it as a record of the original environment rather than a current dependency lock. In particular, the research pipeline is not verified for modern PyTorch, Apple Silicon/MPS, or current Python releases, so it is not expected to run unchanged on a modern Mac.

See [`RESEARCH_CODE.md`](RESEARCH_CODE.md) for the verified limitations, filename history, and what would be required to restore reproducible training or custom-melody inference.

## Dataset

The experiments use the [Hooktheory Lead Sheet Dataset](https://github.com/wayne391/lead-sheet-dataset), which contains event-based JSON, MIDI files, chord symbols, and Roman-numeral labels. The prepared data used by the historical scripts is not included in this repository, and the original external download is no longer available.

Large datasets, checkpoints, NumPy arrays, and generated results are excluded by `.gitignore` so they are not accidentally added to Git history.

## Repository layout

```text
docs/                 Static web sample explorer
hparams/              Historical experiment configurations
model/                CVAE and SurpriseNet model definitions
tests/                Web catalog and asset checks
utils/                Data, decoding, and evaluation utilities
eval.py               Incomplete historical evaluation artifact
train.py              Incomplete historical training artifact
tonal.py              Tonal-space utilities
```

## Paper

> Yi-Wei Chen, Hung-Shin Lee, Yen-Hsing Chen, and Hsin-Min Wang. “SurpriseNet: Melody Harmonization Conditioning on User-controlled Surprise Contours.” Proceedings of the 22nd International Society for Music Information Retrieval Conference, 2021.

```bibtex
@inproceedings{chen2021surprisenet,
  title     = {SurpriseNet: Melody Harmonization Conditioning on User-controlled Surprise Contours},
  author    = {Chen, Yi-Wei and Lee, Hung-Shin and Chen, Yen-Hsing and Wang, Hsin-Min},
  booktitle = {Proceedings of the 22nd International Society for Music Information Retrieval Conference},
  year      = {2021}
}
```

## License

This repository is distributed under the terms in [`LICENSE`](LICENSE).
