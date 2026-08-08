# SurpriseNet

SurpriseNet is a user-controlled conditional variational autoencoder for melody harmonization. It conditions generated chord progressions on a surprise contour selected by the listener.

[Web sample explorer](https://scmvp301135.github.io/SurpriseNet/) · [Paper](https://arxiv.org/abs/2108.00378) · [ISMIR 2021](https://ismir2021.ismir.net/)

## Web sample explorer

The [web demo](https://scmvp301135.github.io/SurpriseNet/) contains precomputed samples. It does not run model inference and does not require a backend.

## Research code status

The Python research code is preserved as an incomplete historical snapshot. It does not provide reproducible preprocessing, supported end-to-end training or custom-melody inference, or a pretrained checkpoint. See [`RESEARCH_CODE.md`](RESEARCH_CODE.md) for verified limitations and restoration requirements.

## Dataset

The experiments use the [Hooktheory Lead Sheet Dataset](https://github.com/wayne391/lead-sheet-dataset), which contains event-based JSON, MIDI files, chord symbols, and Roman-numeral labels. The prepared data used by the historical scripts is not included in this repository, and the original external download is no longer available.

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
