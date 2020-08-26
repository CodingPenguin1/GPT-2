# GPT-2

## Requirements

`gpt-2-simple==1.7.0` (I installed through pip)
`tensorflow-gpu==1.5.0` or `tensorflow==1.5.0` (I installed through conda)

## Usage

Save your datasets in `datasets`. See examples in there for formatting. `datasets/r6_op_bios` shows how to combine multiple datasets.

Train by running `./train_model.py datasets/DATASET STEPS`, where STEPS is the number of generations to run, divided by 1000. For example, passing 10 will run 10k generations.

Run your trained model with `./run_model.py MODEL_NAME "PROMPT"` where MODEL_NAME is the directory under `checkpoint` of the model you would like to use. PROMPT is an optional string that gives the model a prompt of what to write about.
