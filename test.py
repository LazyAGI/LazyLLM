# from lazyllm import WebModule

# def func(x):
#     return '# Test\nHere is a local image:\n\n![Temperature](/home/mnt/chenzhe1/WorkDir/images/temperature_chart.png) \n ' \
#            'Here is an online image: ![](https://i-blog.csdnimg.cn/direct/93410ddf60da438aa362d87b3d33d72f.jpeg)'

# WebModule(func, port=12365, title="Test Markdown",).start().wait()

import lazyllm
chat = lazyllm.OnlineChatModule(model='SenseNova-V6-5-Pro', source='sensenova')
lazyllm.WebModule(chat, port=range(23466, 23470), stream=True).start().wait()