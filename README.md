# SurpriseNet

### SurpriseNet: Melody Harmonization Conditioning on User-controlled Surprise Contours
This is the source code of SurpriseNet, a user-controlled conditional CVAE model based on user's indication to complete melody harmonization.
Some generated samples are available at [https://scmvp301135.github.io/SurpriseNet](https://scmvp301135.github.io/SurpriseNet).

For more information, see our paper:
[arXiv paper](https://arxiv.org/abs/2108.00378).

### Installation
To install SurpriseNet, clone the repo and install it using conda:

```
# First clone and enter the repo
git clone https://github.com/scmvp301135/SurpriseNet.git
cd SurpriseNet
```

* Create environment with conda:
```
# First clone and enter the repo
conda env create -f environment.yml
conda activate surprisenet
```

### Downloading Dataset

We performed experiments on the [Hooktheory Lead Sheet Dataset (HLSD)](https://github.com/wayne391/lead-sheet-dataset) , which contains high-quality and human-arranged melodies with chord progressions. The dataset is provided in two formats, event-based JSON files and MIDI files. Furthermore, there are many types of labels on chords, such as chord symbols and Roman numerals for reference. You have to download the dataset first to reproduce the work.

```
git clone https://github.com/wayne391/lead-sheet-dataset

cd lead_sheet_dataset/src
python theorytab_crawler.py # Remember to uncomment Line 12 in theorytab_crawler.py to crawl full songs. 
python main.py

```

### Convert data to numpy for training 

After downloading, we have to convert JSON files to npy and npz files and for training. 

### Create surprise contours data

After converting, we have to create surprise contours data and weight chord data for training as well.

### Training
All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
pip install -r requirements.txt
python surprisenet_train.py
```

`surprisenet_train.py` is written using argparse

```bash
python surprisenet_train.py with -epoch 10 -save_model model_surprisenet
```

Trained models will be saved in the specified `save_model` which is a required argument.

### Inference

All package requirements are contained in `requirements.txt`. To train the model, run:

```bash
python surprisenet_inference.py
```

`surprisenet_inference` is also written using argparse, give `model_path` to generate chords:

```bash
python surprisenet_inference.py with -model_path model_surprisenet
```

### Interative Demo Website

Coming soon...






