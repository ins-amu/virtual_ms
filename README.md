# The Virtual Multiple Sclerosis Patient

## Introduction
This repository contains the code and data accompanying the paper **"The Virtual Multiple Sclerosis Patient"**. 

Sorrentino, P., Pathak, A., Ziaeemehr, A., Lopez, E.T., Cipriano, L., Romano, A., Sparaco, M., Quarantelli, M., Banerjee, A., Sorrentino, G. and Jirsa, V., 2024. The virtual multiple sclerosis patient. [iScience](https://www.cell.com/iscience/fulltext/S2589-0042(24)01326-9), 27(7).


The work was supported by the EBRAINS Italy nodo Italiano grant, CUP B51E22000150006, and by the European Union’s Horizon 2020 research and innovation program under grant agreement No. 101147319 (EBRAINS 2.0 Project), No. 101137289 (Virtual Brain Twin Project), and ANR-22-PESN-0012 (France 2030 program). A.P. and A.B. were supported by NBRC core funds. A.B. was supported by Ministry of Youth Affairs and Sports, Government of India, Award ID: F.NO. K-15015/42/2018/SP-V, NBRC Flagship program, Department of Biotechnology, Government of India, Award ID: BT/MED-III/NBRC/Flagship/Flagship2019.



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
