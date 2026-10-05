#!/bin/bash
#SBATCH --job-name=layout_opt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Default project root is the repository root (parent of jobs/).
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-layout_opt}"

if [[ "$#" -eq 0 ]]; then
	echo "Usage: sbatch jobs/slurm_job_template.sh <command> [args ...]"
	exit 1
fi

mkdir -p "${PROJECT_ROOT}/logs"

# Activate the project environment.
# Replace this line if your cluster uses a different conda init path.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"

# Example command pattern:
# python Dual_barcodes/generate_dual_layout.py --config Dual_barcodes/config_dual_set1.json
# python Quadtree/generate_layouts_118x182_hybrid.py
# python Zipcodes/generate_layouts_118x182_zipcodes.py
# python HSDB/layout_generator/dual_barcode_hierarchical.py --array-rows 118 --array-cols 182 --bar1 HSDB/bar1_zipcode/bar1_barcodes_28nt.txt --out-dir HSDB/outputs/

echo "Running in ${PROJECT_ROOT}: $*"
"$@"

printf '\nJob finished successfully at %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
