# The Virtual Multiple Sclerosis Patient

## Introduction
This repository contains the code and data accompanying the paper **"The Virtual Multiple Sclerosis Patient"**. .

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)

## Installation

```bash
# create conda env 
conda create -n vms_env python=3.10
conda activate vms_env

git clone https://github.com/ins-amu/virtual_ms.git
cd virtual_ms
pip install -r requirements.txt
pip install -e .
```

## Usage
To see the usage please look at the notebooks of [forward_modeling.ipynb](https://github.com/ins-amu/virtual_ms/blob/main/notebooks/forward_modeling.ipynb) and [inverse_modeling.ipynb](https://github.com/ins-amu/virtual_ms/blob/main/notebooks/inverse_modeling.ipynb).
