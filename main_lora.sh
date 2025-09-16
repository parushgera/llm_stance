#!/bin/bash -l
#SBATCH -o std_out_main_lora_mistral_7b_llama3_8b_phi
#SBATCH -e std_err_main_lora_mistral_7b_llama3_8b_phi
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:2
#SBATCH --mail-user=parush@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

source activate llm_stance
python /home/p/parush/llm_stance/main_lora.py --model mistral_7b llama3_8b phi