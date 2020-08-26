#!/usr/bin/env python

import gpt_2_simple as gpt2
from sys import argv
import os
from shutil import rmtree


def download_model(model_name):
    if not os.path.isdir(os.path.join('models', model_name)):
        print(f'Downloading {model_name} model')
        gpt2.download_gpt2(model_name=model_name)


if __name__ == '__main__':
    models = ['355M', '345M', '124M', '117M']

    dataset_file = argv[1]
    if len(argv) <= 2:
        steps = 1000
    else:
        steps = int(argv[2])
        if steps < 1000:
            steps *= 1000

    session = gpt2.start_tf_sess()
    for model in models:
        download_model(model)

        model_name = dataset_file[dataset_file.rfind('/') + 1:].replace('.txt', '') + '-' + model

        try:
            message = f'# TRAINING MODEL {model} ({steps} steps)#'
            print(len(message) * '#')
            print(message)
            print(len(message) * '#')
            gpt2.finetune(session,
                          dataset=dataset_file,
                          model_name=model,
                          steps=steps,
                          restore_from='fresh',
                          run_name=model_name,
                          print_every=10,
                          sample_every=10000000000,
                          save_every=100)
            break
        except Exception:
            message = f'# MODEL {model} FAILED TO RUN #'
            print(len(message) * '#')
            print(message)
            print(len(message) * '#')
            rmtree(f'checkpoint/{model_name}')
