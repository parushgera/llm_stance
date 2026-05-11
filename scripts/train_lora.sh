#!/bin/bash -l
#SBATCH -o std_out_train_lora
#SBATCH -e std_err_train_lora
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:2
#SBATCH --mail-user=parush@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

source activate llm_stance
# Default: train all four base models. Pass --models phi mistral_7b ... to subset.
python scripts/train_lora.py "$@"
