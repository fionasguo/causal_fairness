#!/bin/bash
#SBATCH --partition=donut-default
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50GB

source ~/anaconda3/bin/activate damf_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/nas/home/siyiguo/anaconda3/lib

# simply finetune bert
# python src/finetune_bert.py --train_path ~/causal_fairness/data/train_data.csv --test_path ~/causal_fairness/data/test_data_relaxed.csv -o ./model_outputs_test_on_relaxed_1

# fair bert
# python src/fair_bert.py --mode train --train_path ~/causal_fairness/data/train_data.csv --test_path ~/causal_fairness/data/test_data_relaxed.csv -o ./model_outputs_test_on_relaxed_1

python src/fair_bert.py --mode test --test_path ~/causal_fairness/data/test_data_original.csv --model_path ~/causal_fairness/model_outputs_test_on_original/best_model -o ./test_on_original
python src/fair_bert.py --mode test --test_path ~/causal_fairness/data/test_data_relaxed.csv --model_path ~/causal_fairness/model_outputs_test_on_original/best_model -o ./test_on_relaxed
python src/fair_bert.py --mode test --test_path ~/causal_fairness/data/test_data_strict.csv --model_path ~/causal_fairness/model_outputs_test_on_original/best_model -o ./test_on_strict

# rpryzant's causal bert
# python src/causal_bert_others.py --train_path ~/causal_fairness/data/train_data.csv --test_path ~/causal_fairness/data/test_data_relaxed.csv -o ./cb_model_outputs_on_relaxed