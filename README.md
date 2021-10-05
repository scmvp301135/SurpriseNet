# SurpriseNet

### SurpriseNet: Melody Harmonization Conditioning on User-controlled Surprise Contours
This is the source code of SurpriseNet, a user-controlled conditional CVAE model based on user's indication to complete melody harmonization.
Some generated samples are available at [https://scmvp301135.github.io/SurpriseNet](https://scmvp301135.github.io/SurpriseNet).

For more information, see our paper:
[arXiv paper](https://arxiv.org/abs/2108.00378).

### Installation
* To install SurpriseNet, clone the repo and install it using conda:

```
# First clone and enter the repo
git clone https://github.com/scmvp301135/SurpriseNet.git
cd SurpriseNet
```

* Create environment with conda:
```
conda env create -f environment.yml
conda activate surprisenet
```

### Downloading Dataset

We performed experiments on the [Hooktheory Lead Sheet Dataset (HLSD)](https://github.com/wayne391/lead-sheet-dataset) , which contains high-quality and human-arranged melodies with chord progressions. The dataset is provided in two formats, event-based JSON files and MIDI files. Furthermore, there are many types of labels on chords, such as chord symbols and Roman numerals for reference. 

‼️ We recommend downloading the prepared dataset directly from the link below, as the crawler program is not ready for the updated website.

***Latest Update:***
***Sample Dataset: 2018/8/1***
***Source:*** [Link](https://drive.google.com/file/d/13iB5Brk1hypKsw9TSf8_d4Ka3xU0XmFZ/view?usp=sharing) (4.9 G).  

* Or use wget to download  google drive files:
```
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/u/0/uc?id=13iB5Brk1hypKsw9TSf8_d4Ka3xU0XmFZ&export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/u/0/uc?id=13iB5Brk1hypKsw9TSf8_d4Ka3xU0XmFZ&export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O hooktheory_dataset.tar.gz && rm -rf /tmp/cookies.txt

tar -xvf hooktheory_dataset.tar.gz
```

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






