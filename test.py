from lazyllm.tools.fs import FeishuFS
f = FeishuFS(user_refresh_token='auto', space_id='7242199391661080580')
with f.open('/大模型工具链/研发流程', 'rb') as fp:
    text = fp.read().decode('utf-8')

text_v2 = text + '\n\n--- 以上内容已于 2026-03-17 审阅 ---'
data_v2 = text_v2.encode('utf-8')
import tempfile, os
with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
    tmp.write(data_v2)
    tmp_path = tmp.name

f.put_file(tmp_path, '/大模型工具链/研发流程_v2')
print(f.ls('/大模型工具链/'))
