# Installation Guide

## Requirements

- Python 3.9
- conda or mamba
- a Linux environment for Slurm execution if you want to submit jobs to a cluster

## Set up the environment

From the repository root:

```bash
conda env create -f env/environment.yml
conda activate layout_opt
```

## Verify the environment

```bash
python -V
python -c "import numpy, pandas, scipy, matplotlib; print('environment ok')"
```

## Run a generator

```bash
python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
```

## Optional: install the package in editable mode

If your cluster or workflow requires package installation from the local directory:

```bash
pip install -e Zipcodes
```

This is optional and depends on how the project is being used in your environment.
