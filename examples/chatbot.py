import lazyllm

t = lazyllm.TrainableModule('internlm2-chat-7b')
lazyllm.WebModule(t, port=23466).start().wait()
