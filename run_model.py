#!/usr/bin/env python

import gpt_2_simple as gpt2
from sys import argv

if __name__ == '__main__':
    run_name = argv[1]
    truncate = None
    if len(argv) <= 2:
        prompt = ''
    elif len(argv) <= 3:
        prompt = argv[2]
    else:
        prompt = argv[2]
        truncate = argv[3]

    session = gpt2.start_tf_sess()
    print(run_name)
    gpt2.load_gpt2(session, run_name=run_name)
    gpt2.generate(session, run_name=run_name, prefix=prompt, truncate=truncate, nsamples=5, batch_size=5)
