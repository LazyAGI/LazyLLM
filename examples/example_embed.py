import lazyllm
from lazyllm import deploy

base_embed = '/mnt/lustrenew/share_data/lazyllm/models/bge-large-zh-v1.5'
m = lazyllm.TrainableModule(base_embed).deploy_method(deploy.AutoDeploy)

m.evalset(['你好', '世界'])
m.update_server().eval()

print(m.eval_result)
