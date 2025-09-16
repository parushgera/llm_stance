#!/bin/bash -l
#SBATCH -o std_out_main_in_target_lora
#SBATCH -e std_err_main_in_target_lora
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:2
#SBATCH --mail-user=parush@usf.edu # email for notifications
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE # events for notifications

source activate llm_stance
python /home/p/parush/llm_stance/main_in_target_v2.py --use-lora --lora-dir /storage3-ciber/parush/lora


