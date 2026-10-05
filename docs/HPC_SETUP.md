# HPC and Slurm Setup

This repository is designed to run on Slurm-based HPC systems.

## Job template

The job template is located at:

- jobs/slurm_job_template.sh

It is written to:

- activate the conda environment
- set up a log directory
- run a Python script passed as an argument

## Submit a job

```bash
sbatch jobs/slurm_job_template.sh python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
```

To run Dual for different dimensions without editing source code:

```bash
sbatch jobs/slurm_job_template.sh python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json --rows 253 --cols 363
```

If the script requires arguments:

```bash
sbatch jobs/slurm_job_template.sh python HSDB/layout_generator/dual_barcode_hierarchical.py --array-rows 118 --array-cols 182 --bar1 HSDB/bar1_zipcode/235x363_extra/bar1_barcodes_28nt.txt --out-dir HSDB/outputs/
```

## Output logs

The template writes stdout and stderr files into the logs directory.

## Recommended practice

- keep each generated output in a dedicated directory per run
- avoid mixing source files with generated CSV and barcode outputs
- use the environment file as the canonical dependency specification
