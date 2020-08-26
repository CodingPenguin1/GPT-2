#!/usr/bin/env python

import gpt_2_simple as gpt2
from sys import argv

if __name__ == '__main__':
    run_name = argv[1]
    if len(argv) <= 2:
        prompt = ''
    else:
        prompt = argv[2]

    session = gpt2.start_tf_sess()
    print(run_name)
    gpt2.load_gpt2(session, run_name=run_name)
    gpt2.generate(session, run_name=run_name, prefix=prompt)
