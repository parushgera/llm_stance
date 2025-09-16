#!/bin/bash -l
#SBATCH -o std_out_parsing_llama3_8b
#SBATCH -e std_err_parsing_llama3_8b
#SBATCH -p SIPEIE23
#SBATCH -w GPU48
#SBATCH --mem=200GB
#SBATCH --gres=gpu:1
#SBATCH --mail-user=parush@usf.edu
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE

source activate llm_stance
python /home/p/parush/llm_stance/parsing.py --model llama3_8b --no-clarify


