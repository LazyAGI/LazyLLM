import lazyllm

chat = lazyllm.TrainableModule('internlm2-chat-7b')

if __name__ == '__main__':
    lazyllm.WebModule(chat, port=23466).start().wait()
