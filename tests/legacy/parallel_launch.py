import os

os.environ["LAZYLLM_REDIS_URL"] = "redis://SH-IDC1-10-142-4-31:9997"
# os.environ['LAZYLLM_REDIS_URL'] = ''

import lazyllm

t1 = (
    lazyllm.TrainableModule(stream=False)
    .finetune_method(lazyllm.finetune.dummy)
    .deploy_method(lazyllm.deploy.dummy)
    .mode("finetune")
    .prompt("hello world1 <{input}>")
)
prompter = lazyllm.Prompter(
    prompt="hello world2 <{input}>, hisory is <{history}>",
    history_symbol="history",
    eoh="[EOH]",
    eoa="[EOA]",
)
t2 = (
    lazyllm.TrainableModule(stream=True)
    .finetune_method(lazyllm.finetune.dummy)
    .deploy_method(lazyllm.deploy.dummy)
    .mode("finetune")
    .prompt(prompter)
)

t3 = (
    lazyllm.TrainableModule(stream=False)
    .finetune_method(lazyllm.finetune.dummy)
    .deploy_method(lazyllm.deploy.dummy)
    .mode("finetune")
    .prompt(prompter)
)

s0 = lazyllm.ServerModule(lazyllm.pipeline(t1, t2, t3), stream=True)

w = lazyllm.WebModule(
    s0,
    port=20584,
    components={
        t1: [("do_sample", "Checkbox", True), ("temperature", "Text", 0.1)],
        t2: [("do_sample", "Checkbox", False), ("temperature", "Text", 0.2)],
    },
)

t1 = lazyllm.ForkProcess(target=w.update, args=(), sync=False)
t1.start()
t1.join()
