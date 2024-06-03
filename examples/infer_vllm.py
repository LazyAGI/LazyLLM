import lazyllm
from lazyllm import launchers, deploy

base_model = '/mnt/lustrenew/share/qitianlong/models/internlm2-chat-1_8b'
m = lazyllm.TrainableModule(base_model, '').deploy_method(deploy.vllm, launcher=launchers.remote(ngpus=1))

dataset = ['介绍一下你自己', '李白和李清照是什么关系', '说个笑话吧']
m.evalset([f'<|im_start|>user\n{x}<|im_end|>\n<|im_start|>assistant\n' for x in dataset])

m.update_server()
m.eval()
print(m.eval_result)
