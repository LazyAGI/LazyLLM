import os
import multiprocessing

os.environ['LAZYLLM_DEBUG'] = '1'
os.environ['LAZYLLM_LOG_FILE_MODE'] = "merge"  # default 'merge'

import lazyllm
global_variable = 0


def worker(num):
    global global_variable
    global_variable += num
    import time
    time.sleep(5)
    lazyllm.LOG.error(f"Process global_variable = {global_variable}")


if __name__ == "__main__":
    processes = []
    for i in range(5):
        p = multiprocessing.get_context('fork').Process(target=worker, args=(i,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(f"Main process: global_variable = {global_variable}")
    lazyllm.LOG.complete()