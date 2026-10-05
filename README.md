# Barcode Layout Methods for Spatial Transcriptomics Arrays

This repository contains a compact, shareable version of the barcode-generation project used to design spatially aware DNA arrays for transcriptomics experiments.

## Algorithms included

### Dual barcodes
The dual-barcode strategy partitions the array into blocks and assigns a shared block-level identifier plus a shorter position-specific identifier. The idea is to preserve local spatial structure while keeping the full barcode globally unique.

### Quadtree
The quadtree-based method encodes spatial coordinates hierarchically using recursive subdivision. This produces barcodes whose differences reflect spatial proximity, which is useful when neighboring spots should remain distinguishable but similar.

### Zipcodes
The zipcode approach uses spatially ordered code assignments, where nearby locations receive similar barcode patterns. This is useful for maintaining a structured distance relationship across the array while preserving sequence constraints.

### HSDB
HSDB combines the dual-barcode idea with the zipcode concept: block identifiers are generated with spatial organization, and within-block spot identifiers are kept compact and reusable. This gives a structured, hierarchical layout while simplifying the per-block decoding problem.

## Repository structure

- Dual_barcodes/: dual-barcode generator, libraries, and config templates
- Quadtree/: quadtree generator, libraries, and config templates
- Zipcodes/: zipcode generator and config templates
- HSDB/: hierarchical spatial dual-barcode implementation
- env/: conda environment definition
- jobs/: Slurm job templates
- docs/: installation and HPC instructions

## Installation

Create and activate the conda environment from the repository root:

```bash
conda env create -f env/environment.yml
conda activate layout_opt
```

If you are using a Python virtual environment instead of conda, make sure it contains the packages listed in the environment file.

## Config templates

Start from the template file for the method you want to run:

- [Dual_barcodes/config_dual_template.json](Dual_barcodes/config_dual_template.json)
- [Quadtree/config_quadtree_template.json](Quadtree/config_quadtree_template.json)
- [Zipcodes/config_zipcodes_template.json](Zipcodes/config_zipcodes_template.json)

Copy a template to a new config file and edit the fields you need: array dimensions, library paths, output paths, and any method-specific parameters.

## Running locally

From the repository root:

```bash
python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
python Quadtree/generate_quadtree_layout.py --config Quadtree/config_quadtree_set1.json
python Zipcodes/generate_zipcode_layout.py --config Zipcodes/config_zipcodes_set1.json
python HSDB/layout_generator/dual_barcode_hierarchical.py --array-rows 118 --array-cols 182 --bar1 HSDB/bar1_zipcode/235x363_extra/bar1_barcodes_28nt.txt --out-dir HSDB/outputs/
```

For a different Dual layout size, override the dimensions and optionally the output prefix:

```bash
python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json --rows 253 --cols 363 --output-prefix Dual_barcodes/custom_run_253x363
```

## Running on Slurm

Use the reusable job template in the jobs directory:

```bash
sbatch jobs/slurm_job_template.sh python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
sbatch jobs/slurm_job_template.sh python Quadtree/generate_quadtree_layout.py --config Quadtree/config_quadtree_set1.json
sbatch jobs/slurm_job_template.sh python Zipcodes/generate_zipcode_layout.py --config Zipcodes/config_zipcodes_set1.json
```

For scripts that require arguments:

```bash
sbatch jobs/slurm_job_template.sh python HSDB/layout_generator/dual_barcode_hierarchical.py --array-rows 118 --array-cols 182 --bar1 HSDB/bar1_zipcode/235x363_extra/bar1_barcodes_28nt.txt --out-dir HSDB/outputs/
```

## Notes

- This copy is intentionally compact and keeps only the essential runnable code and setup.
- Historical benchmark and analysis artifacts were intentionally omitted from this public-facing copy.

