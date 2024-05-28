import sys
sys.path.append('..')

import lazyllm
from lazyllm import dataproc

@lazyllm.llmregister('dataproc')
def gen_data(idx):
    print(f'idx {idx}: gen data done')
    return idx + 1

@lazyllm.llmregister('dataproc')
def gen_data2(idx1, idx2):
    print(f'idx {idx1}, {idx2}: gen data done')
    return idx1 + idx2

ppl = lazyllm.pipeline(
    dataproc.gen_data(),
    lazyllm.parallel(
        lazyllm.pipeline(
            dataproc.gen_data(),
            lazyllm.barrier,
            dataproc.gen_data(),
            lazyllm.barrier,
            dataproc.gen_data(),
            lazyllm.parallel(
                lazyllm.pipeline(
                    dataproc.gen_data(),
                    lazyllm.barrier,
                    dataproc.gen_data(),
                ),
                lazyllm.pipeline(
                    dataproc.gen_data(),
                    lazyllm.barrier,
                    dataproc.gen_data(),
                ),
            ),
            dataproc.gen_data2(),
            dataproc.gen_data(),
            dataproc.gen_data(),
        ),
        lazyllm.pipeline(
            dataproc.gen_data(),
            lazyllm.barrier,
            dataproc.gen_data(),
            dataproc.gen_data(),
            dataproc.gen_data(),
            dataproc.gen_data(),
            lazyllm.barrier,
            dataproc.gen_data(),
        ),
    ),
    dataproc.gen_data2(),
    dataproc.gen_data(),
)

ppl(0)