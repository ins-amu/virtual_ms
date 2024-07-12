# The Virtual Multiple Sclerosis Patient

## Introduction
This repository contains the code and data accompanying the paper **"The Virtual Multiple Sclerosis Patient"**. 

Sorrentino, P., Pathak, A., Ziaeemehr, A., Lopez, E.T., Cipriano, L., Romano, A., Sparaco, M., Quarantelli, M., Banerjee, A., Sorrentino, G. and Jirsa, V., 2024. The virtual multiple sclerosis patient. [iScience](https://www.cell.com/iscience/fulltext/S2589-0042(24)01326-9), 27(7).

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
Please refere to the notebooks: 
- [forward_modeling.ipynb](https://github.com/ins-amu/virtual_ms/blob/main/notebooks/forward_modeling.ipynb) 
- [inverse_modeling.ipynb](https://github.com/ins-amu/virtual_ms/blob/main/notebooks/inverse_modeling.ipynb).
