# Bogazici University Deep Learning Course Assignment 2

This repository contains all the work that [Özgür Deniz Demir](https://github.com/odenizddd) and [Osman Yasin Baştuğ](https://github.com/yasinbastug) has done for the second assignment of the CmpE 597 Deep Learning course given at Bogazici University in Spring 2025.

## 1. Setup

We recommend that you create a virtual environment in order to use this repository.

To do this you can clone the repository and in the root of this project run the following commands:

```bash
python3 -m venv env
source ./env/bin/activate
pip install -r requirements.txt
```

## 2. Run the scripts

We provide several different scripts in order to train different models. To run those scripts first navigate to the scripts directory.

```bash
cd src/scripts
```

Then run one of the following scripts.

### Convolutional AutoEncoder

```bash
python cnnae.py
```

### LSTM AutoEncoder

```bash
python lstmae.py
```
