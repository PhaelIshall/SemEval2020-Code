# Commonsense Validation and Explanation Task
This repository contains the code for our submission to SemEval-2020 task 4: Commonsense Validation and Explanation Task (ComVe). The task is explained in the picture below. Please look at the [paper](https://www.aclweb.org/anthology/2020.semeval-1.51/) here for more details 

The data provided by the task organizers is available under the following [repository](https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation).

Subtask A and B will are evaluated using accuracy. Subtask C is evaluated using BLEU score.

![alt text](https://github.com/PhaelIshall/SemEval2020-Code/blob/master/comve.png "ComVe tasks")

# How to run the code


### Setup
      export COMVE_DIR=/path/to/COMVE_data_dir
      export OUTPUT_DIR=/path/to/output_dir 
      conda create -n comve_env python=3.6
      conda activate comve_env
      pip3 install requirements.txt
        
## Task A   
      python3 ./run_taskA.py \
              --model_type bert \
              --task_name swag \
              --model_name_or_path bert-base-uncased \
              --do_train \
              --do_eval \
              --do_test \
              --data_dir $COMVE_DIR \
              --learning_rate 5e-5 \
              --num_train_epochs 3 \
              --max_seq_length 80 \
              --output_dir $OUTPUT_DIR \
              --per_gpu_eval_batch_size=8 \
              --per_gpu_train_batch_size=8 \
              --gradient_accumulation_steps 2 \
              --overwrite_output > results.csv 

## Task B
      export COMVE_DIR=/path/to/COMVE_data_dir 
      export OUTPUT_DIR=/path/to/output_dir 
      python3 ./run_taskB.py \
      --model_type bert \
      --task_name swag \
      --model_name_or_path bert-base-uncased\
      --do_train \
      --do_eval \
      --do_test \
      --data_dir $COMVE_DIR \
      --learning_rate 5e-5 \
      --num_train_epochs 3 \
      --max_seq_length 80 \
      --output_dir $OUTPUT_DIR \
      --per_gpu_eval_batch_size=8 \
      --per_gpu_train_batch_size=8 \
      --gradient_accumulation_steps 2 \
      --overwrite_output > results.csv
      


# Credits
## Hugging Face Transformer Library 

We use and modify the code provided by Hugging Face library for Multiple Choice Tasks that are publically available. We procide a link to the original code for a SWAG example by Hugging Face [here](https://github.com/huggingface/transformers/tree/master/examples/multiple-choice) and the paper [here](https://arxiv.org/abs/1910.03771)
